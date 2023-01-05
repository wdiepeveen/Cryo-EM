from datetime import datetime
import logging
import matplotlib.pyplot as plt
import mrcfile
import os
import pickle
import time
import zipfile
import numpy as np

logger = logging.getLogger(__name__)


class Exp:
    def __init__(
            self,
            debug_level=None,
            debug_timer_start=None,
            results_folder=None,
    ):
        self.debug_level = debug_level
        self.debug_timer_start = debug_timer_start
        # self.exp_cancelled = exp_cancelled_v
        self.results_folder = results_folder

    def begin(self, aux_folders=None, prefix=None, postfix=None):

        # self.exp_cancelled_v = False
        self.debug_timer_start = 1

        if prefix is not None:
            prefix = prefix + "_"
        else:
            prefix = ""
        now = datetime.now()
        runid = now.strftime("%y-%m-%d_%H-%M-%S")
        if postfix is not None:
            postfix = "_" + postfix
        else:
            postfix = ""

        self.results_folder = os.path.join(os.path.dirname(__file__), "..", "results", prefix + runid + postfix)
        os.makedirs(self.results_folder)

        logger.info("Results folder is {}".format(self.results_folder))

        files_list = []
        if aux_folders is None:
            aux_folders = []
        elif isinstance(aux_folders, str):
            aux_folders = [aux_folders]
        else:
            if not isinstance(aux_folders, list):
                raise RuntimeError("aux_folder is not str or list")

        for folder in aux_folders:
            if not isinstance(folder, str):
                raise RuntimeError("Elements in aux_folder are no strings")
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.endswith(".py"):
                        files_list.append(os.path.join(root, file))

        for file in os.listdir("."):
            if file.endswith(".py"):
                files_list.append(file)

        ZipFile = zipfile.ZipFile(os.path.join(self.results_folder, "src.zip"), "w")

        for file in files_list:
            ZipFile.write(file, compress_type=zipfile.ZIP_DEFLATED)

        self.reset_timer()

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

    def filename(self, filename):
        return os.path.join(self.results_folder, filename)

    def folder(self):
        return self.results_folder

    def reset_timer(self):
        self.debug_timer_start = time.time()

    def open_pkl(self, dir, filename):
        # TODO also right away copy the file to results_folder
        with open(os.path.join(dir, filename + ".pkl"), 'rb') as input:
            return pickle.load(input)

    def open_mrc(self, dir, filename):
        # TODO also right away copy the file to results_folder
        return mrcfile.open(os.path.join(dir, filename + ".mrc"))

    def open_npy(self, dir, filename):
        # TODO also right away copy the file to results_folder
        return np.load(os.path.join(dir, filename + ".npy"))

    def save(self, filename, *args):

        if len(args) == 0:
            raise RuntimeError("Nothing to save")

        global results_prefix
        fn = os.path.join(self.results_folder, filename + ".pkl")

        s = dict()
        for arg in args:
            s["{}".format(arg[0])] = arg[1]

        with open(fn, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(s, output, pickle.HIGHEST_PROTOCOL)

    def save_mrc(self, filename, volume, voxel_size=None):

        with mrcfile.new(os.path.join(self.results_folder, filename + ".mrc"), overwrite=True) as mrc:
            mrc.set_data(volume)
            if voxel_size is not None:
                mrc.voxel_size = voxel_size

    def save_mrcs(self, filename, volume, voxel_size=None):

        with mrcfile.new(os.path.join(self.results_folder, filename + ".mrcs"), overwrite=True) as mrc:
            mrc.set_data(volume)
            if voxel_size is not None:
                mrc.voxel_size = voxel_size

    def save_npy(self, filename, data):
        np.save(os.path.join(self.results_folder, filename + ".npy"), data)

    def save_fig(self, filename, save_eps=False):

        plt.savefig(os.path.join(self.results_folder, filename + ".png"))

        if save_eps:
            plt.savefig(os.path.join(self.results_folder, filename + ".eps"))

    def save_im(self, filename, im):
        plt.imsave(os.path.join(self.results_folder, filename + ".png"), im, cmap=plt.gray())

    def save_table(self, filename, values, headers=None, side_headers=None):

        rows, columns = values.shape
        if columns > 1:
            if side_headers is None:
                begin_table = r"\begin{tabular}" + "{}".format("{" + "c|" * (columns - 1) + "c" + "}") + "\n"
            else:
                begin_table = r"\begin{tabular}" + "{}".format("{" + "c|" * columns + "c" + "}") + "\n"
        else:
            if side_headers is None:
                begin_table = r"\begin{tabular}{c}" + "\n"
            else:
                begin_table = r"\begin{tabular}{c|c}" + "\n"

        double_hline = r"\hline\hline" + "\n"

        table_data = ""

        if headers is not None:
            if side_headers is not None:
                table_data += side_headers[0] + " & "

            for col in range(columns):
                table_data += "{}".format(headers[col])
                if col == columns - 1:
                    table_data += r" \\ \hline" + "\n"
                else:
                    table_data += " & "

        for row in range(rows):
            if side_headers is not None:
                table_data += side_headers[row + 1] + " & "
            for col in range(columns):

                table_data += "{}".format(values[row, col])
                if col == columns - 1:
                    if row != rows - 1:
                        table_data += r" \\ \hline" + "\n"
                    else:
                        table_data += r" \\" + "\n"
                else:
                    table_data += " & "

        end_table = r"\end{tabular}"

        table = begin_table + double_hline + table_data + double_hline + end_table

        with open(os.path.join(self.results_folder, filename + ".txt"), "w") as input:
            input.write(table)

    def dbg(self, level, message):

        if self.debug_level is not None or level <= self.debug_level:
            if self.debug_timer_start is not None:
                if level == 0:
                    logger.info("{}".format(message))
                else:
                    logger.info("[{:9.4f}] {}".format(time.time() - self.debug_timer_start, message))

    def dbglevel(self, level):
        self.debug_level = level

    def dbglevel_atleast(self, level):
        return self.debug_level is not None or level <= self.debug_level
