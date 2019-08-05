import os
import sys
import logging
import traceback
from datetime import datetime
import argparse

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='', dest='command')

    # create subparsers:
    # a = subparsers.add_parsers("<command name>")
    # a.add_argument(...)

    args = parser.parse_args()

    # checking if the checkpoints directory exists in the output folder.
    # if it does, then resumes training.
    if not os.path.exists(os.path.join(args.out_dir, "checkpoints")):
        args.out_dir = os.path.join(args.out_dir, datetime.now().strftime("%d_%m_%Y_%H_%M_%S") +
                                    ('-' + args.exp_name if args.exp_name != '' else ''))
        args.checkpoint_dir = os.path.join(args.out_dir, "checkpoints/")
        os.makedirs(args.out_dir)
        os.makedirs(args.checkpoint_dir)
        os.makedirs(os.path.join(args.checkpoint_dir, "best/"))
    else:
        args.checkpoint_dir = os.path.join(args.out_dir, "checkpoints/")

    # creating logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    fmt = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s", "%d-%m-%Y %H:%M:%S")

    file_handler = logging.FileHandler(os.path.join(args.out_dir, "log"))
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    if args.verbose:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)
    
    sys.excepthook = lambda tp, val, tb: logger.error("Unhandled Exception:\nType: %s\nValue: %s\nTraceback: %s\n",
                                                      tp, val, ''.join(traceback.format_tb(tb)))
    
    # logging args
    logging.info("Program started.")
    logging.info("Arguments: %s", args)
    logging.info("Logging results every %d epoch(s)", args.log_freq)

    # checking if cuda is available but is not being used
    if torch.cuda.is_available() and not args.cuda:
        msg = "You have a CUDA device, so you should probably run with --cuda"
        if not args.verbose:
            print("WARNING " + msg)
        logger.warning(msg)

    # executing
    # execute commands as follows:
    # if args.command = <command name>:
    #     do stuff...
