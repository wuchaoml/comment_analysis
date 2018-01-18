import datetime
import time
import tensorflow as tf
import config
import os
from data_utils import Data
from char_cnn import CharConvNet

if __name__ == '__main__':
    print('start...')
    tf.set_random_seed(0.18)
    exec(open('config.py').read())  # 加载配置文件
    print(config.model.th)
    print('end...')
    print('Loading data...')
    train_data = Data(max_length=config.max_length,
                      batch_size=config.batch_size, no_of_classes=config.no_of_classes)
    train_data.build_model_dataset('train')
    dev_data = Data(max_length=config.max_length,
                    batch_size=config.batch_size, no_of_classes=config.no_of_classes)
    dev_data.build_model_dataset('dev')

    '''epoch计算'''
    num_batches_per_epoch = int(
        train_data.get_length() / config.batch_size) + 1
    num_batch_dev = dev_data.get_length()
    print('Loaded')

    print('Training')

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True,  # 如果指定的设备不存在，允许TF自动分配设备
                                      log_device_placement=False)  # 不打印设备分配日志
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            char_cnn = CharConvNet(conv_layers=config.model.conv_layers,
                                   fully_layers=config.model.fully_connected_layers,
                                   max_length=config.max_length,
                                   no_of_classes=config.no_of_classes,
                                   th=config.model.th)
            global_step = tf.Variable(0, trainable=False)

            optimizer = tf.train.AdamOptimizer(config.learning_rate)
            '''可以使用minimize()函数取代这两步'''
            grads_and_vars = optimizer.compute_gradients(
                char_cnn.loss)      # 计算梯度，默认对所有的Variable计算梯度
            train_op = optimizer.apply_gradients(
                grads_and_vars, global_step=global_step)  # 梯度更新

            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram(
                        '{}/grad/hist'.format(v.name), g)
                    sparsity_summary = tf.summary.scalar(
                        '{}/grad/sparsity'.format(v.name), tf.nn.zero_fraction(g))  # tf.nn.zero_fraction()是返回在g中0的比例
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)

            grad_summaries_merged = tf.summary.merge(grad_summaries)

            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(
                os.path.curdir, 'runs', timestamp))
            print('Writing to {}\n'.format(out_dir))

            loss_summary = tf.summary.scalar('loss', char_cnn.loss)
            acc_summary = tf.summary.scalar('accuracy', char_cnn.accuracy)

            train_summary_op = tf.summary.merge(
                [loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
            train_summary_writer = tf.summary.FileWriter(
                train_summary_dir, sess.graph_def)

            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, 'summaries', 'dev')
            dev_summary_writer = tf.summary.FileWriter(
                dev_summary_dir, sess.graph_def)

            checkpoint_dir = os.path.abspath(
                os.path.join(out_dir, 'checkpoints'))
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables())

            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                feed_dict = {char_cnn.input_x: x_batch, char_cnn.input_y: y_batch,
                             char_cnn.dropout_keep_prob: config.training.p,
                             char_cnn.train_test_flag: False,
                             char_cnn.step: global_step}

                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, char_cnn.loss, char_cnn.accuracy], feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print('{}:step {},loss{:g},acc{:g}'.format(
                    time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                feed_dict = {char_cnn.input_x: x_batch,
                             char_cnn.input_y: y_batch, char_cnn.dropout_keep_prob: 1.0,
                             char_cnn.train_test_flag: True,
                             char_cnn.step: global_step}
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, char_cnn.loss, char_cnn.accuracy], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print('{}:step {},loss {:g}, acc {:g}'.format(
                    time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            for e in range(config.training.epoches):
                print(e)
                train_data.shuffle_data()
                for k in range(num_batches_per_epoch):
                    batch_x, batch_y = train_data.get_batch_data(k)
                    train_step(batch_x, batch_y)
                    current_step = tf.train.global_step(sess, global_step)

                    if current_step % config.training.evaluate_every == 0:
                        dev_data.shuffle_data()
                        xin, yin = dev_data.get_batch_data()
                        print('\nEvaluation:')
                        dev_step(xin, yin, writer=dev_summary_writer)
                        print('')

                    if current_step % config.training.checkpoint_every == 0:
                        path = saver.save(
                            sess, checkpoint_prefix, global_step=current_step)
                        print('Saved model checkpoint to {}\n'.format(path))
