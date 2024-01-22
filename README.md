# CS304-SpeechRecognition
The group project by Jingheng Huan and Gezhi Wang

## Description

This project is designed to capture, process, and visualize audio data, specifically focusing on the extraction and plotting of Mel-Frequency Cepstral Coefficients (MFCC). It includes a set of Python scripts that handle audio capture, MFCC computation, and visualization of audio signals and features.

## Installation

Before running the project, ensure you have Python installed on your system. You will also need the following libraries:
- numpy
- scipy
- matplotlib
- wave

Install these using pip:
```bash
pip install numpy scipy matplotlib wave
```

## Usage

To use this project, run the `main.py` script:
```bash
python main.py
```
This will start the audio capture process, compute the MFCC, and plot the waveform, spectrogram, and cepstrum, saving each plot as an image.

## Files and Functions

- `audio_capture.py`: Handles the audio recording functionality.
- `audio_utils.py`: Contains utility functions for audio processing.
- `config.py`: Configuration settings for the project.
- `main.py`: The main script that orchestrates the capture and plotting process.
- `plotting.py`: Contains functions to plot the audio waveform, spectrogram, and cepstrum.

## Contributing

If you'd like to contribute to this project, please fork the repository and create a pull request with your features or fixes.

## License

This project is licensed under the MIT License - see the `LICENSE.md` file for details.

## Acknowledgments
