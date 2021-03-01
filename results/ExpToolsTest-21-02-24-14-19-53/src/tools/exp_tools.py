from datetime import datetime
import logging
import matplotlib.pyplot as plt
import mrcfile
import os
import pickle
import time
import zipfile

# TODO do we need this?
logger = logging.getLogger(__name__)


def exp_begin(aux_folders = "", prefix = "", postfix = ""):

    global results_folder
    # global results_prefix
    global exp_cancelled_v

    exp_cancelled_v = False

    if not prefix == "":
        prefix =  prefix + "-"
    now = datetime.now()
    runid = now.strftime("%y-%m-%d-%H-%M-%S")
    if not postfix == "":
        postfix = "-" + postfix

    results_folder = os.path.join("results", prefix + runid + postfix)
    os.makedirs(results_folder)

    # results_prefix = results_folder + "/"


    files_list = []
    if aux_folders == "":
        aux_folders = []
    elif isinstance(aux_folders,str):
        aux_folders = [aux_folders]
    else:
        if not isinstance(aux_folders,list):
            raise RuntimeError("aux_folder is not str or list")

    for folder in aux_folders:
        if not isinstance(folder, str):
            raise RuntimeError("Elements in aux_folder are no strings")
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith(".py"):
                    files_list.append(os.path.join(root, file))
                    print(os.path.join(root, file))
                # files.append(folder + "/*.py")

    for file in os.listdir("."):
        if file.endswith(".py"):
            files_list.append(file)
            print(file)


    ZipFile = zipfile.ZipFile(os.path.join(results_folder, "src.zip"), "w")

    for file in files_list:
        print(file)
        ZipFile.write(file, compress_type=zipfile.ZIP_DEFLATED)
        # ZipFile.write(file)

    exp_reset_timer()


def exp_cancelled():
  global exp_cancelled_v
  return exp_cancelled_v


# TODO exp_comment
# function exp_comment(onwhat::String = "")
# 	global io
# 	global results_prefix
#
# 	if onwhat == ""
# 		prefix = ""
# 	else
# 		prefix = onwhat * ": "
# 	end
#
# 	print("Comment: ")
# 	msg = readline()
# 	@printf io "%s\n" "Comment: " * msg
#
# 	fid  = open(results_prefix * "_comment.txt", "w+")
# 	write(fid, prefix * msg * "\n")
# 	close(fid)
#
# end

# TODO exp_end
# def exp_end():
    # TODO stop timer here or so?


def exp_filename(filename):
    global results_folder
    return os.path.join(results_folder, filename)


def exp_folder():
    global results_folder
    return results_folder


def exp_reset_timer():
    global debug_timer_start
    debug_timer_start = time.time()


# def exp_save_object(obj, filename):
#     with open(filename, 'wb') as output:  # Overwrites any existing file.
#         pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def exp_open(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)


# TODO exp_save (and del save_object)
def exp_save(filename, *args):
    global results_folder

    if len(args)==0:
        raise RuntimeError("Nothing to save")

    global results_prefix
    fn = os.path.join(results_folder, filename + ".pkl")

    s = dict()
    for arg in args:
        s["{}".format(arg)] = arg

    with open(fn, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(s, output, pickle.HIGHEST_PROTOCOL)


def exp_save_mrc(filename,volume):

    global results_folder

    with mrcfile.new(os.path.join(results_folder, filename + ".mrc"), overwrite=True) as mrc:
        mrc.set_data(volume)


def exp_savefig(filename, figure, save_eps = False):
    global results_folder

    figure.savefig(os.path.join(results_folder, filename + ".png"))

    if save_eps:
        figure.savefig(os.path.join(results_folder,  filename + ".eps"))


# TODO exp_save_im
# function exp_saveim(img::Array{K,2}, filename::String) where K
# 	global results_folder
#
# 	save(results_prefix * filename * ".png",img)
# end

# TODO exp_save_table
# function exp_savetable(table::String, filename::String) where K
# 	global results_folder
#
# 	io = open(results_prefix * filename * ".txt","w")
# 	write(io,table)
# 	close(io)
# end
