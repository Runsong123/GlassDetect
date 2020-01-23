import json
def check_json_with_error_msg(pred_json, num_classes=10):
    '''
    Args:
        pred_json (str): Json path
        num_classes (int): number of foreground categories
    Returns:
        Message (str)
    Example:
        msg = check_json_with_error_msg('./submittion.json')
        print(msg)
    '''
    if not pred_json.endswith('.json'):
        return "the prediction file should ends with .json"
    with open(pred_json) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return "the prediction data should be a dict"
    if not 'images' in data:
        return "missing key \"images\""
    if not 'annotations' in data:
        return "missing key \"annotations\""
    images = data['images']
    annotations = data['annotations']
    if not isinstance(images, (list, tuple)):
        return "\"images\" format error"
    if not isinstance(annotations, (list, tuple)):
        return "\"annotations\" format error"
    for image in images:
        if not 'file_name' in image:
            return "missing key \"file_name\" in \"images\""
        if not 'id' in image:
            return "missing key \"id\" in \"images\""
    for annotation in annotations:
        if not 'image_id' in annotation:
            return "missing key \"image_id\" in \"annotations\""
        if not 'category_id' in annotation:
            return "missing key \"category_id\" in \"annotations\""
        if not 'bbox' in annotation:
            return "missing key \"bbox\" in \"annotations\""
        if not 'score' in annotation:
            return "missing key \"score\" in \"annotations\""
        if not isinstance(annotation['bbox'], (tuple, list)):
            return "bbox format error"
        if len(annotation['bbox'])==0:
            return "empty bbox"
        if annotation['category_id'] > num_classes or annotation['category_id'] < 0:
            return "category_id out of range"
    return "right"

if __name__ == "__main__":
	# parser = argparse.ArgumentParser(description="Generate result")
	# parser.add_argument("-m", "--model",help="Model path",type=str,)
	# parser.add_argument("-c", "--config",help="Config path",type=str,)
	# parser.add_argument("-im", "--im_dir",help="Image path",type=str,)
	# parser.add_argument('-o', "--out",help="Save path", type=str,)
	# args = parser.parse_args()
	# model2make_json = args.model
	# config2make_json = args.config
	# json_out_path = args.out
	# pic_path = args.im_dir
	# result_from_dir()
	print(check_json_with_error_msg("/Users/zhuzhuxia/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/10011886514ee26da8fdca7bcb65e20f/Message/MessageTemp/94916fd49fe8e1b7ebbf7e596b58c3c1/File/20200118_yong_baseline78.json"))