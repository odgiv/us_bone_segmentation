import numpy as np
from PIL import Image


def read_us_data(filename):
    with open(filename, 'rb') as fid:

        data_array = np.fromfile(fid, np.uint8)  # data_array is 1D np array

        # The header size is 19 ints. Note, an integer in C has a size of 4
        # bytes. The header contains the width, height and depth at positions
        # 2, 3, and 1, respectively (in terms of C storage order).
        header = data_array[0:(4*19)]
        depth = (header[5] << 8) + header[4]
        width = (header[9] << 8) + header[8]
        height = (header[13] << 8) + header[12]
        image = data_array[4*19:]

        image_data = np.reshape(image, (height, width, depth), order='F')

        fid.close()

        return {
            'filename': filename,
            'header': header,
            'image_data': image_data,
            'width': width,
            'height': height,
            'depth': depth
        }


def store_us_data(filename, image_data):
    """
    image_data is 3D numpy array.
    TODO: It is not complete. Finish implementation.
    """
    (height, width, depth) = image_data.shape

    # first 4 bytes are not used
    header = bytearray(np.zeros((4*19,), dtype=int))
    header[4:6] = (height).to_bytes(2, byteorder='big')
    header[8:10] = (width).to_bytes(2, byteorder='big')
    header[12:14] = (depth).to_bytes(2, byorder='big')
    # order='F' is for Fortran ordering
    header.append(image_data.tobytes(order='F'))

    with open(filename, 'wb') as fid:
        fid.write(header)

    fid.close()


if __name__ == "__main__":
    data = read_us_data(
        "D:\\Data\\IPASM\\bone_data\\phantom_data\\juliana wo symImages\\13-51-47\\postProcessedImage.b8")
    # print(data)
    aslice = data["image_data"][:, :, 250]  # as example, take a slice in depth
    print(aslice.shape)
    img = Image.fromarray(aslice)
    img.show()

    # image_data = data["image_data"]
    # store_us_data("D:\\Data\\US Calibration\\2018-11-15 3D phantom training data\\ScanConverted\\11-42-11\\test.b8", image_data)
