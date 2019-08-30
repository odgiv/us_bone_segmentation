import numpy as np
from PIL import Image


def read_us_data(filename):
    """
    Method for reading US file with .b8 ext.
    """
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


if __name__ == "__main__":
    data = read_us_data("D:\\Data\\IPASM\\bone_data\\phantom_data\\juliana wo symImages\\13-51-47\\postProcessedImage.b8")    
    aslice = data["image_data"][:, :, 250]  # as example, take a slice in depth
    print(aslice.shape)
    img = Image.fromarray(aslice)
    img.show()