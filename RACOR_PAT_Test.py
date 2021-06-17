import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from utils.metrics_loss import model_loss, PSNR, SSIM, KLDivergence, SavingMetric
from gooey import Gooey, GooeyParser


@Gooey(program_name='Dataset preparation')
def main():
    setting_msg = 'Simulated Data '
    parser = GooeyParser(description=setting_msg)
    parser.add_argument('test_dir', help='test directory',type=str, widget='FileChooser',default='')
    parser.add_argument('model_dir', help='nae of the figure to be saved', type=str, widget='DirChooser', default='')
    parser.add_argument('num_samples', help='number of samples to plot ',type=int,  default= 3)
    parser.add_argument('im_no', help='range of images ', type=int, default=0)
    parser.add_argument('fig_name', help='name of the figure', type=str, default='testplot')
    parser.add_argument('save_fig', help='do you want to save figure True/False', default=False)
    args = parser.parse_args()
    return args.test_dir,args.model_dir,args.num_samples,args.im_no,args.fig_name,args.save_fig


def data_gen(filename):
    path = tf.keras.utils.get_file(filename,origin=None)
    with np.load(path) as data:
        src_images = data['arr_0']
        tar_images = data['arr_1']
    return src_images,tar_images

def model_load(test_filename,model_dir,num_samples,im_no,fig_name,save_fig):
    predictions_test = list()
    src_test , tar_test = data_gen(test_filename)
    print(len(src_test))
    test_dataset = tf.data.Dataset.from_tensor_slices((src_test))
    test_dataset = test_dataset.batch(1)
    filename = sorted(os.listdir(model_dir))[-1]
    directory = os.path.join(os.getcwd(), model_dir)
    directory = os.path.join(directory, filename)
    saved_model = tf.keras.models.load_model(directory, custom_objects={'model_loss':model_loss,'loss_func':model_loss(),'PSNR':PSNR, 'SSIM':SSIM, 'KLDivergence':KLDivergence,'SavingMetric':SavingMetric})
    print('Done Loading Best Model(' + filename + ') from: ' + model_dir)
    for element in test_dataset.as_numpy_iterator():
        predictions_curr = saved_model.predict(element, steps = 1)
        predictions_test.append(predictions_curr)
    [predictions_test] = [np.asarray(predictions_test)]
    predictions = np.reshape(predictions_test, (predictions_test.shape[0],128, 128))
    src_images = np.reshape(src_test, (src_test.shape[0],128, 128))
    tar_images = np.reshape(tar_test, (tar_test.shape[0],128, 128))
    for i in range(num_samples):
        plt.subplot(3, num_samples, 1 +  i)
        plt.axis('off')
        plt.imshow(src_images[i+im_no],cmap='gist_yarg')
        plt.title('input')
        plt.subplot(3, num_samples, 1 +num_samples+ i)
        plt.axis('off')
        plt.imshow(predictions[i+im_no],cmap='gist_yarg')
        plt.title('predicted')
        plt.subplot(3, num_samples, 1 + num_samples*2  + i)
        plt.axis('off')
        plt.imshow(tar_images[i+im_no],cmap='gist_yarg')
        plt.title('ground truth')
    if save_fig is True:
        plt.savefig(fig_name+'.jpg',dpi=150)
    plt.show()


if __name__=='__main__':
    test_dir, model_dir,num_samples, im_no, fig_name, save_fig = main()
    model_load(test_dir, model_dir, num_samples, im_no, fig_name, save_fig)
