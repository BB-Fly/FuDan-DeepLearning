import numpy as np
import struct


def decode_idx3(idx3_file):
    bin_data = open(idx3_file, 'rb').read()
    offset = 0
    fmt_header = '>iiii'
    _, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)

    image_size = num_rows*num_cols

    offset += struct.calcsize(fmt_header)
    fmt_image = '>'+str(image_size)+'B'
    images = np.empty((num_images, num_rows,num_cols))

    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image,bin_data, offset)).reshape(num_rows,num_cols)
        offset += struct.calcsize(fmt_image)

    return images

def decode_idx1(idx1_file):
    bin_data = open(idx1_file, 'rb').read()
    offset = 0
    fmt_header = '>ii'
    _, num_images = struct.unpack_from(fmt_header, bin_data, offset)

    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    images = np.empty(num_images)

    for i in range(num_images):
        images[i] = struct.unpack_from(fmt_image,bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)

    return images
        