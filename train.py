import time
import numpy as np 
import tensorflow as tf
from tensorflow.contrib import slim
from nets import model
from data import icdar
from nets.config import Config
from nets.model  import build_SPC, build_input_graph
from data import icdar

tf.app.flags.DEFINE_integer('num_readers', 10, '')
tf.app.flags.DEFINE_string('training_data_path', 'data/icdar2017/', '')
tf.app.flags.DEFINE_float('learning_rate', 0.002, '')
tf.app.flags.DEFINE_integer('decay_steps', 10000, '')
tf.app.flags.DEFINE_integer('max_steps', 300000, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'models/icdar2017/', '')
tf.app.flags.DEFINE_boolean('restore', True, 'whether to resotre from checkpoint')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 10000, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_string('pretrained_model_path', "models/pretrained_models/resnet_v1_50.ckpt", '')
tf.app.flags.DEFINE_integer('display_steps', 20, '')

FLAGS = tf.app.flags.FLAGS
config = Config()

def main(argv=None):
	import os
	os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
	if not tf.gfile.Exists(FLAGS.checkpoint_path):
		tf.gfile.MkDir(FLAGS.checkpoint_path)
	# build network input graph
	train_input = build_input_graph(is_training=True, config=config)
	global_step = tf.train.get_or_create_global_step()
	learning_rate = tf.train.polynomial_decay(FLAGS.learning_rate, global_step,\
					decay_steps=FLAGS.decay_steps, end_learning_rate=1e-5, power=0.9, cycle=True)
	opt = tf.train.AdamOptimizer(learning_rate)
	losses = build_SPC(train_input, config, is_training = True)
	grads = opt.compute_gradients(losses['total_loss'])
	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
	batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
	# save moving average
	variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
	variables_averages_op = variable_averages.apply(tf.trainable_variables())
	# batch norm updates
	with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
		train_op = tf.no_op(name='train_op')
	saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
	init = tf.global_variables_initializer()

	if FLAGS.pretrained_model_path is not None:
		variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path, \
								tf.get_collection(tf.GraphKeys.MODEL_VARIABLES),\
								ignore_missing_vars=True)
	gpu_options = tf.GPUOptions(allow_growth = True)
	with tf.Session(config = tf.ConfigProto(allow_soft_placement = True, gpu_options = gpu_options)) as sess:
		ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
		if FLAGS.restore and ckpt != None:
			print 'continue training from previous checkpoint', ckpt
			saver.restore(sess, ckpt)
		else:
			sess.run(init)
			if FLAGS.pretrained_model_path is not None:
				variable_restore_op(sess)
				print 'Restore from  pretrained model ', FLAGS.pretrained_model_path
		# Generate input data
		data_generator = icdar.get_batch(num_workers = FLAGS.num_readers,\
										 data_path = FLAGS.training_data_path, cfg=config)

		start = time.time()
		for step in range(0, FLAGS.max_steps+1):
			data = next(data_generator)
			feed_dict = {}
			for key in data.keys():
				feed_dict[train_input[key]] = data[key]
			_lr, _global_step, _losses, _ = sess.run([learning_rate, global_step, \
												losses, train_op], feed_dict=feed_dict)
			'''
			for key in _losses.keys():
				print key, _losses[key]
			'''
			if np.isnan(_losses['total_loss']):
				print("Loss diverged !!!")
				break

			if step % FLAGS.display_steps == 0:
				time_per_step = (time.time() - start) / FLAGS.display_steps
				start = time.time()
    	
				print('''Step {}  |  learning_rate = {:.6f} |  Total Loss = {:.4f}  |  {:.2f} s/step
				RPN loss  |  class_loss = {:.4f}  bbox_loss = {:.4f}  
				TCM loss  |  global_mask_loss = {:.4f}
				MRCNN loss|  class_loss = {:.4f}  bbox_loss = {:.4f}  mask_loss = {:.4f}'''
						.format(int(_global_step), _lr, _losses['total_loss'], time_per_step,
								 	_losses['rpn_class_loss'], _losses['rpn_bbox_loss'], 
									_losses['global_mask_loss'], _losses['mrcnn_class_loss'], _losses['mrcnn_bbox_loss'],
									_losses['mrcnn_mask_loss']))

			if step % FLAGS.save_checkpoint_steps == 0:
				saver.save(sess, FLAGS.checkpoint_path + 'model.ckpt', global_step = global_step)

if __name__ == '__main__':
	tf.app.run()
