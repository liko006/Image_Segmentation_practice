import os
import numpy as np
from PIL import Image

__all__ = ['get_color_pallete', 'print_iou', 'set_img_color',
           'show_prediction', 'show_colorful_images', 'save_colorful_images']


def print_iou(iu, mean_pixel_acc, class_names=None, show_no_back=False):
    n = iu.size
    lines = []
    for i in range(n):
        if class_names is None:
            cls = 'Class %d:' % (i + 1)
        else:
            cls = '%d %s' % (i + 1, class_names[i])
        # lines.append('%-8s: %.3f%%' % (cls, iu[i] * 100))
    mean_IU = np.nanmean(iu)
    mean_IU_no_back = np.nanmean(iu[1:])
    if show_no_back:
        lines.append('mean_IU: %.3f%% || mean_IU_no_back: %.3f%% || mean_pixel_acc: %.3f%%' % (
            mean_IU * 100, mean_IU_no_back * 100, mean_pixel_acc * 100))
    else:
        lines.append('mean_IU: %.3f%% || mean_pixel_acc: %.3f%%' % (mean_IU * 100, mean_pixel_acc * 100))
    lines.append('=================================================')
    line = "\n".join(lines)

    print(line)


def set_img_color(img, label, colors, background=0, show255=False):
    for i in range(len(colors)):
        if i != background:
            img[np.where(label == i)] = colors[i]
    if show255:
        img[np.where(label == 255)] = 255

    return img


def show_prediction(img, pred, colors, background=0):
    im = np.array(img, np.uint8)
    set_img_color(im, pred, colors, background)
    out = np.array(im)

    return out


def show_colorful_images(prediction, palettes):
    im = Image.fromarray(palettes[prediction.astype('uint8').squeeze()])
    im.show()


def save_colorful_images(prediction, filename, output_dir, palettes):
    '''
    :param prediction: [B, H, W, C]
    '''
    im = Image.fromarray(palettes[prediction.astype('uint8').squeeze()])
    fn = os.path.join(output_dir, filename)
    out_dir = os.path.split(fn)[0]
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    im.save(fn)


def get_color_pallete(npimg, dataset='pascal_voc'):
    """Visualize image.
    Parameters
    ----------
    npimg : numpy.ndarray
        Single channel image with shape `H, W, 1`.
    dataset : str, default: 'pascal_voc'
        The dataset that model pretrained on. ('pascal_voc', 'ade20k')
    Returns
    -------
    out_img : PIL.Image
        Image with color pallete
    """
    # recovery boundary
    if dataset in ('pascal_voc', 'pascal_aug'):
        npimg[npimg == -1] = 255
    # put colormap
    if dataset == 'ade20k':
        npimg = npimg + 1
        out_img = Image.fromarray(npimg.astype('uint8'))
        out_img.putpalette(adepallete)
        return out_img
    elif dataset == 'citys':
        out_img = Image.fromarray(npimg.astype('uint8'))
        out_img.putpalette(cityspallete)
        return out_img
    out_img = Image.fromarray(npimg.astype('uint8'))
    out_img.putpalette(vocpallete)
    return out_img
