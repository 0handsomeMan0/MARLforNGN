import json

def write_log(filename, data):
    with open('logs/' + filename, 'a') as file:
        file.write(str(data) + '\n')


def write_json_log(filename, data):
    with open('logs/' + filename, 'a') as f:
        json.dump(data, f)