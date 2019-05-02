import numpy as np 
import tensorflow as tf
from tensorflow.contrib import slim
import cv2
from nets import utils
from nets.config import Config
from nets.model  import build_SPC, build_input_graph
import os

tf.app.flags.DEFINE_string('test_data_path', 'data/demo/', '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'models/icdar2017/', '')
tf.app.flags.DEFINE_string('output_dir', 'data/demo/results', None)
tf.app.flags.DEFINE_bool('write_images', True, 'write images')

FLAGS = tf.app.flags.FLAGS

class InferenceConfig(Config):
	# Set batch size to 1 since we'll be running inference on
	# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	DETECTION_MIN_CONFIDENCE = 0
config = InferenceConfig()

# Read .bmp .jpg .gif
def read_image(file):
	im = cv2.imread(file)
	if im is None:
		gif = imageio.mimread(file)
		if gif is not None:
			gif = np.array(gif)
			gif = gif[0]
			im = gif[:, :, 0:3].astype(np.float32)
	else:
		im = im.astype(np.float32)
	return im

def get_model_list(model_path=FLAGS.checkpoint_path):
    ckpt=tf.train.get_checkpoint_state(model_path)
    last_ckpt = tf.train.latest_checkpoint(model_path)
    ckpt_path_list = []
    if ckpt is not None:
      ckpt_path_list = ckpt.all_model_checkpoint_paths
    return ckpt_path_list, last_ckpt

def get_image_list(data_path = FLAGS.test_data_path):
	assert os.path.exists(data_path),"This path is not exists!"
	if os.path.isfile(data_path):
		return [data_path]
	else:
		image_list = []
		for file in os.listdir(data_path):
			fn = os.path.join(data_path, file)
			if os.path.isfile(fn) and \
				file.split('.')[-1] in ['bmp', 'jpg', 'gif', 'png']:
				image_list.append(fn)
		return image_list

def get_result(model_path, image_list):
	os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
	tf.reset_default_graph()
	with tf.get_default_graph().as_default():
		# build the whole network
		# input: [1, config.IMAGE_SHAPE[0]. config.IMAGE_SHAPE[1], 3]
		test_input = build_input_graph(is_training=False, config=config)
		# detections: [1, num_detections, (y1, x1, y2, x2, class_id, score)]
		# mask: [1, num_detections, config.MASK_SHAPE[0], config.MASK_SHAPE[1], 2]
		detections, masks = build_SPC(test_input, config, is_training=False)
		# Restore the SPCNet
		global_step = slim.get_or_create_global_step()
		variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
		saver = tf.train.Saver(variable_averages.variables_to_restore())
		gpu_options = tf.GPUOptions(allow_growth = True)
		with tf.Session(config = tf.ConfigProto(allow_soft_placement = True, gpu_options = gpu_options)) as sess:
			print 'Restore from {}'.format(model_path)
			saver.restore(sess, model_path)
			# detect single image
			# batch_size > 1 not implement
			for ix, im_fn in enumerate(image_list):
				print model_path+"  "+str(ix+1)+'/'+str(len(image_list))
				image = read_image(im_fn)
				if image is None:
					print imfn, " is empty !"
					continue
				# Resize image
				molded_image, window, scale, padding, crop = utils.resize_image(image,
							min_dim=config.IMAGE_MIN_DIM,
							min_scale=config.IMAGE_MIN_SCALE, 
							max_dim=config.IMAGE_MAX_DIM,
							mode=config.IMAGE_RESIZE_MODE)
				_detections, _masks = sess.run([detections, masks,gts],
							feed_dict={test_input['input_image'] : molded_image[np.newaxis,...]})
				# Inference Batch size = 1
				final_boxes, final_class_ids, final_scores, final_masks, bound_boxes = \
							utils.unmold_detections(_detections[0], _masks[0],
							image.shape, molded_image.shape, window)
				print "Detect %d text in %s " % (bound_boxes.shape[0], im_fn)
				if FLAGS.write_images and os.path.exists(FLAGS.output_dir):
					save_image_path = os.path.join(FLAGS.output_dir, os.path.basename(im_fn))
					result_image = image
					
					for i,box in enumerate(final_boxes):
						y1, x1, y2, x2 = box
						cv2.rectangle(result_image,(x1,y1),(x2,y2),(0,255,0),1)
					
					for box in bound_boxes:
						for i in range(4):
							p1 = (box[2*i],box[2*i+1])
							p2 = (box[(i+1)%4*2], box[(i+1)%4*2+1])
							cv2.line(result_image, p1, p2, (255,0,0), 2)
					
					if cv2.imwrite(save_image_path, result_image): # Image save successfully
						print "Save detected results in ",save_image_path

if __name__ == "__main__":
	model_list, last_model= get_model_list()
	image_list = get_image_list()
	if len(image_list) > 0:
		get_result(last_model, image_list)
