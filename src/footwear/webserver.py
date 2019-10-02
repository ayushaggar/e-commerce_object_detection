import numpy as np
import requests
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import plot_util
from object_detection.utils import label_map_util
import object_detection.utils.ops as utils_ops
from PIL import Image

import json
import logging
import os
from pkg_resources import resource_filename
import time

from flask import (Flask, request, url_for, render_template, abort,
                   send_from_directory)
import numpy as np
from skimage.io import imread

app = Flask(__name__)
TEMP_MEDIA_FOLDER = os.path.join(os.getcwd(), 'flask_media')
app.config['UPLOAD_FOLDER'] = TEMP_MEDIA_FOLDER
TEMP_RESULT_FOLDER = os.path.join(os.getcwd(), 'flask_result')
app.config['RESULT'] = TEMP_RESULT_FOLDER
app.config['SERVER_URL'] = "http://localhost:8501/v1/models/footwear:predict"
app.config['SAVE_OUTPUT_IMAGE'] = "True"
app.config['PATH_TO_LABEL'] = os.path.join(os.getcwd(), 'data_label_path')

@app.route('/footwear/img_upload', methods=['GET', 'POST'])
def upload_file():
    return render_template('img_upload.html')


@app.route('/footwear/result', methods=['POST'])
def process_img_upload():
    """
    Handles HTTP request by saving and detected image.

    Returns:
        method call to `show_footwear_success` that calls to render our results'
        template with the request result
    """
    footwear_result = {}
    request_timestamp = int(time.time()*1000)
    footwear_result['timestamp'] = request_timestamp

    img_file = request.files['image']
    if '.jpg' not in img_file.filename:
        abort(415, "No .jpg-file provided")

    # create a new filename and store image
    img_filename = 'img_upload_' + str(request_timestamp) + '.jpg'
    img_filepath = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
    img_file.save(img_filepath)

    # URL of the tensorflow-serving Docker. 
    server_url = app.config['SERVER_URL']
    
    # Path to jpeg image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
    
    # Path to output json file resulting from the API call
    img_filename_output = 'img_result_json_' + str(request_timestamp) + '.json'
    output_image = os.path.join(app.config['RESULT'], img_filename_output)
    
    # Whether output_image should be save from the predictions
    save_output_image = app.config['SAVE_OUTPUT_IMAGE']

    # Path to label map, which is json-file that maps each category name to a unique number
    path_to_labels = os.path.join(app.config['PATH_TO_LABEL'], "object-detection.pbtxt")

    footwear_result['img_filename'] = url_for('get_file', filename=img_filename_output)

    # Build input data
    formatted_json_input = pre_process(image_path)
    print('Pre-processing done! \n')

    # Call tensorflow server
    headers = {"content-type": "application/json"}
    print('\n\nMaking request to {server_url}...\n')
    server_response = requests.post(server_url, data=formatted_json_input, headers=headers)
    print('Request returned\n')

    # Post process output
    print('\n\nPost-processing server response...\n')
    image = Image.open(image_path).convert("RGB")
    image_np = load_image_into_numpy_array(image)
    output_dict = post_process(server_response, image_np.shape)
    print('Post-processing done!\n')

    # Save output on disk
    print('\n\nSaving output to {output_image}\n\n')
    with open(output_image, 'w+') as outfile:
        json.dump(json.loads(server_response.text), outfile)
    print('Output saved!\n')

    if save_output_image:
        # Save output on disk
        category_index = label_map_util.create_category_index_from_labelmap(path_to_labels, use_display_name=True)

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8,
            )
        output_with_no_extension = output_image.split('.', 1)[0]
        output_image = ''.join([output_with_no_extension, '.jpeg'])
        Image.fromarray(image_np).save(output_image)
        print('\n\nImage saved\n\n')
        
    
    return show_footwear_result(footwear_result)


@app.route('/footwear/img_upload/<path:filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename,
                               as_attachment=True)


def show_footwear_result(footwear_result):
    """
    Handles successful object detection
    and returns the render_template for Flask

    Args:
        footwear_result (dict): Request processing and result information

    Returns:
        (:obj:`flask.render_template`)
    """
    return render_template('result.html', **footwear_result)


def format_mask(detection_masks, detection_boxes, N, image_size):
    """
    Format the m*m detection soft masks as full size binary masks. 
    Args:
        detection_masks (np.array): of size N * m * m
        detection_boxes (np.array): of size N * 4 with the normalized bow coordinates.
            Coordinates are written as [y_min, x_min, y_max, x_max]
        N (int): number of detections in the image
        image_size (tuple(int))
    Returns:
        detection_masks (np.array): of size N * H * W  where H and W are the image Height and Width.
    
    """
    (height, width, _) = image_size
    output_masks = np.zeros((N, image_size[0], image_size[1]))
    # Process the masks related to the N objects detected in the image
    for i in range(N):
        normalized_mask = detection_masks[i].astype(np.float32)
        normalized_mask = Image.fromarray(normalized_mask, 'F')

        # Boxes are expressed with 4 scalars - normalized coordinates [y_min, x_min, y_max, x_max]
        [y_min, x_min, y_max, x_max] = detection_boxes[i]

        # Compute absolute boundary of box
        box_size = (int((x_max - x_min) * width), int((y_max - y_min) * height)) 

        # Resize the mask to the box size using LANCZOS appoximation
        resized_mask = normalized_mask.resize(box_size, Image.LANCZOS)
        
        # Convert back to array
        resized_mask = np.array(resized_mask).astype(np.float32)

        # Binarize the image by using a fixed threshold
        binary_mask_box = np.zeros(resized_mask.shape)
        thresh = 0.5
        (h, w) = resized_mask.shape

        for k in range(h):
            for j in range(w):
                if resized_mask[k][j] >= thresh:
                    binary_mask_box[k][j] = 1

        binary_mask_box = binary_mask_box.astype(np.uint8)

        # Replace the mask in the context of the original image size
        binary_mask = np.zeros((height, width))
        
        x_min_at_scale = int(x_min * width)
        y_min_at_scale = int(y_min * height)

        d_x = int((x_max - x_min) * width)
        d_y = int((y_max - y_min) * height)

        for x in range(d_x):
            for y in range(d_y):
                binary_mask[y_min_at_scale + y][x_min_at_scale + x] = binary_mask_box[y][x] 
        
        # Update the masks array
        output_masks[i][:][:] = binary_mask

    # Cast mask as integer
    output_masks = output_masks.astype(np.uint8)
    return output_masks

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def pre_process(image_path):
    """
    Pre-process the input image to return a json to pass to the tf model
    Args:
        image_path (str):  Path to the jpeg image
    Returns:
        formatted_json_input (str)
    """

    image = Image.open(image_path).convert("RGB")
    image_np = plot_util.load_image_into_numpy_array(image)

    # Expand dims to create  bach of size 1
    image_tensor = np.expand_dims(image_np, 0)
    formatted_json_input = json.dumps({"signature_name": "serving_default", "instances": image_tensor.tolist()})

    return formatted_json_input


def post_process(server_response, image_size):
    response = json.loads(server_response.text)
    output_dict = response['predictions'][0]

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'])
    output_dict['detection_classes'] = np.array([int(class_id) for class_id in output_dict['detection_classes']])
    output_dict['detection_boxes'] = np.array(output_dict['detection_boxes'])
    output_dict['detection_scores'] = np.array(output_dict['detection_scores'])

    # Process detection mask
    if 'detection_masks' in output_dict:
        # Determine a threshold above which we consider the pixel shall belong to the mask
        # thresh = 0.5
        output_dict['detection_masks'] = np.array(output_dict['detection_masks'])
        output_dict['detection_masks'] = format_mask(output_dict['detection_masks'], output_dict['detection_boxes'], output_dict['num_detections'], image_size)
    
    return output_dict


def main():
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == '__main__':
    main()
