#!/usr/bin/python
# -*- coding: utf-8 -*-

from pathlib import Path
mod_suffix = ('.cpp', '.h')
import os
statement = """#pragma execution_character_set("utf-8")\n"""
statement_len = len(statement)

src_encode = "utf-8"
dst_encode = "utf-8"

def is_in_suf_list(name: Path):
    if name.suffix in mod_suffix:
        return True
    return False

# insert statement at the first line of each file whose extension is in mod_suffix


def add_state(dir: Path):
    for x in dir.iterdir():
        # print(x)
        if x.is_dir():
            print("Entering directory: ", str(x))
            add_state(x)
            print("Leaving directory: ", str(x))
        elif x.is_file() and is_in_suf_list(x):
            print(" --modified: ", x)
            with x.open(mode="r", encoding=src_encode) as f:
                raw = f.read()
                s = raw.find(statement)
                if s > -1:
                    # avoid dunplicate
                    continue
            
            with x.open(mode="w", encoding=dst_encode) as f:
                # f.seek(0)
                f.write(statement + raw)


def remove_first_line(dir: Path):
    for x in dir.iterdir():
        # print(x)

        if x.is_dir():
            print("Entering directory: ", str(x))
            remove_first_line(x)
            print("Leaving directory: ", str(x))
        elif x.is_file() and is_in_suf_list(x):
            mod = None
            with x.open(mode="r", encoding=src_encode) as f:
                raw = f.read()
                s = raw.find(statement)
                if s > -1:
                    mod = raw[s+statement_len:]
                # print(s+statement_len, mod)
                # exit(0)
            if mod is None:
                continue
            print(" --modified: ", x)
            with x.open(mode="w", encoding=dst_encode) as f:
                f.write(mod)
                # exit(0)

def change_encoding(dir: Path):
    for x in dir.iterdir():
        # print(x)

        if x.is_dir():
            print("Entering directory: ", str(x))
            change_encoding(x)
            print("Leaving directory: ", str(x))
        elif x.is_file() and is_in_suf_list(x):
            try:
                with x.open(mode="r", encoding=src_encode) as f:
                    raw = f.read()
                    # print(s+statement_len, mod)
                    # exit(0)
                print(f" --modified: {x}, src: {src_encode} => {dst_encode}")
                with x.open(mode="w", encoding=dst_encode) as f:
                    f.write(raw)
            except Exception as ex:
                print(ex, f"file: {x}")

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="relative path to apply actions.")
    parser.add_argument("-a", "--add", help="add statement(s)", action="store_true")
    parser.add_argument("-r", "--remove", help="remove statement(s)", action="store_true")
    parser.add_argument("-e", "--encoding", help="change encoding of file(s)", action="store_true")
    parser.add_argument("--src_encode", help="source file encoding (default utf-8)")
    parser.add_argument("--dst_encode", help="destination file encoding (default utf-8)")
    args = parser.parse_args()
    
    if args.src_encode:
        src_encode = args.src_encode
    if args.dst_encode:
        dst_encode = args.dst_encode
    
    path = Path(args.path)

    if not path.exists():
        print("Error: Path not exists.")
        exit(1)
    if args.remove:
        print("removing")
        remove_first_line(path)
    elif args.add:
        print("inserting")
        add_state(path)
    elif args.encoding:
        print("change encoding")
        change_encoding(path)
    else:
        print(parser.print_help())

