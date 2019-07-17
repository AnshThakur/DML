import tensorflow as tf
import os
import sys
import argparse
import facenet
import numpy as np
from datetime import datetime
import scipy.io as sio
def main(args):
    with tf.Graph().as_default():

        with tf.Session() as sess:

            learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

            image_placeholder = tf.placeholder(tf.float32, shape=(None,40,200,1), name='image')

            dynamic_alpha_placeholder = tf.placeholder(tf.float32, shape=(), name='dynamic_alpha_placeholder')

            input_map = {'image': image_placeholder, 'phase_train': phase_train_placeholder, 'learning_rate': learning_rate_placeholder, 'dynamic_alpha_placeholder': dynamic_alpha_placeholder}
            facenet.load_model(args.model, input_map=input_map)
            # Get output tensor
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            
            main_folder='./1_channel_final/train/'
            folder=next(os.walk(main_folder))[1]
            
            embeddings_array=[]
            Labels=[]
            for i in range(0,71):
                    print(folder[i])
                    print(i)
                    train_files=os.listdir('./1_channel_final/train/'+folder[i])
          
                    for s in range(0,len(train_files)):
                        current_file = np.load('./1_channel_final/train/'+folder[i]+'/'+train_files[s])               
                        #print(current_file.shape)
                        current_file = np.expand_dims(current_file, axis=0)
                        current_file = np.expand_dims(current_file, axis=3)
                        #current_pose = np.expand_dims(current_pose, axis=3)
                        feed_dict = {image_placeholder: current_file}
                        emb = sess.run(embeddings,feed_dict={image_placeholder: current_file})
                        emb=emb.reshape((args.embedding_size,))
                        embeddings_array.append(emb)
                        Labels.append(i)  
                    #os.mkdir('./Embeddings_folds_all_conv/fold'+str(j))
             
            sio.savemat('./Embedding_GHNP/gen.mat',{'embeddings_array':embeddings_array,'Labels':Labels})
                    
            


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str,
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',default='./models/siamese/1channel_inception_v5')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory',default='./folds_1channel')
    parser.add_argument('--number_of_gallery_images', type=int,
        help='Number of images used for training for each subject',default=6)
    parser.add_argument('--embedding_size', type=int,
        help='Dimensionality of the embedding.', default=128)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
