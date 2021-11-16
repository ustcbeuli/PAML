import time


def get_current_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def output_to_file(buf, fout):
    print(buf)
    fout.write(buf + '\n')
    fout.flush()
