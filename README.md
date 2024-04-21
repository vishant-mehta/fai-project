# IntelliGrade: Enhancing Descriptive Answer Assessment with LLMs

## Description
This project contains Python scripts for analyzing the MashQA dataset and augmenting it with additional columns: Answer, Score, and Feedback. The scripts include [csv_json_convert.py](https://github.com/vishant-mehta/fai-project/blob/main/json_csv_convert.py) and [dataset_generator.py](https://github.com/vishant-mehta/fai-project/blob/main/dataset_generator.py), designed to run with minimal dependencies. Additionally, this repository includes a Streamlit application, [user_interface.py](https://github.com/vishant-mehta/fai-project/blob/main/user_interface.py), which serves as a UI dashboard for visualizing results from the underlying trained model.

More details about the project can be found in the [project report](https://github.com/vishant-mehta/fai-project/blob/main/FAI_Project.pdf).


## Installation
Make sure you have Python 3.x installed on your system. You can check your Python version by running:
```bash
python --version
```
Make sure you have installed the following libraries.
```bash
pip install streamlit
```

## Usage

### Running Analysis Scripts
To run the python scripts, navigate to the directory containing the scripts and execute:
```bash
python3 csv_json_convert.py
```
```bash
python3 dataset_generator.py
```

### Running StreamLit application
To run the streamlit application, navigate to the directory containing the streamlit app and execute:
```bash
streamlit run user_interface.py
```

## Contributors

This project has been developed with contributions from the following individuals:

- **Akshat Gandhi** 
- **Vishant Mehta** 
- **Amrit Gupta** 
- **Aswath Senthil Kumar**






