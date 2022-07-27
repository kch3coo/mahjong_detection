from unicodedata import name
import imutils
import cv2
import os

tiles_map = {1: 'bam-1', 2: 'bam-2', 3: 'bam-3', 4: 'bam-4', 5: 'bam-5', 6: 'bam-6', 7: 'bam-7', 8: 'bam-8', 9: 'bam-9', 10: 'char-1', 11: 'char-2', 12: 'char-3', 13: 'char-4', 14: 'char-5', 15: 'char-6', 16: 'char-7', 17: 'char-8', 18: 'char-9', 19: 'dots-1', 20: 'dots-2', 21: 'dots-3', 22: 'dots-4', 23: 'dots-5', 24: 'dots-6', 25: 'dots-7', 26: 'dots-8', 27: 'dots-9', 28: 'h-east', 29: 'h-green', 30: 'h-north', 31: 'h-red', 32: 'h-south', 33: 'h-west', 34: 'h-white'}

def create_diplay_tiles():
    #  get current file path
    current_path = os.getcwd()
    # get file path
    path = os.path.join(current_path, "mahjong_display_source")
    output_path = os.path.join(current_path, "mahjong_display")
    # create output path if not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # get all files in path
    files = os.listdir(path)
    # loop through all files
    for file in files:
        img = cv2.imread(os.path.join(path, file))
        # get file name without extension
        file_name = os.path.splitext(file)[0]
        # get file extension
        file_extension = os.path.splitext(file)[1]
        img = imutils.resize(img, width=40)

        # create output file name
        output_file_name = tiles_map[int(file_name)]
        # create output file path
        output_file_path = os.path.join(output_path, output_file_name + file_extension)
        # save image
        cv2.imwrite(output_file_path, img)
        print("Saved image: " + output_file_name)
if __name__ == '__main__':
    create_diplay_tiles()