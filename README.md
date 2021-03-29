# Facilities for BriFuture

These facility scripts which will mostly be written in python is for better management of myself, but if some of them are also useful for you, you can download them (either source code or install the whole package with pip) for personal usage. It will be grateful if you make some advision or want to make some improvements over these scripts.

## Usable Commands

All commands that is usable can be downloaded and compiled with `pip`, all these commands are prefixed with `bf_` which may make these commands more identified.

- bf_broadcast: Identify which machine running the specified script through sending and recieving broadcast udp datagram.
- bf_gitrepo: which may simplify the creation or deletion a shared git repository on personal mini git server. you can read a blog article on how I make it by visiting [my blog website](http://www.zbrifuture.cn/2019/03/51/).
- bf_monitor: Monitor file changes, and execute prepared commands. you can read a blog article on how I make it by visiting [my blog website](http://www.zbrifuture.cn/2019/03/66/).

For more details about these facilities, please refer the help message provided by each command with `-h` option.

## Install 

Using pip to install these facility commands:

```
python3 -m pip install bffacilities
or
pip install bffacilities
```

## Usage

You can use `bff` or `bffacilities` to call sub commands, for example, let's generate QtTest files from directory:

```sh
## in source directory
cd test
bff gqt -d ../src -p Test -e "test,ui"
```

## ChangeLog

#### v0.0.19
1. some updates on generateQtTest and torch scripts

#### v0.0.17

1. add `generateQtTest` script, to automatically create test file that use `QTest`.

#### v0.0.13

1. add torch scripts and labelme scripts for MachineLearing data processing or model building.

#### v0.0.7

1. add simple tcp socket server

#### v0.0.6

1. add common flask functions

## License

License is GPLv3.