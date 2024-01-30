from config import *
from audio_capture import *
from plotting import *
from audio_utils import *

def main():
    
    digit_to_record = int(input("Input digit to record (0-9):"))
    record_digit(digit_to_record)

if __name__ == '__main__':
    main()