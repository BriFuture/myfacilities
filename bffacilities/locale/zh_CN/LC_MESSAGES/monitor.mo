��          |               �   U   �   1   3  4   e     �  �   �  3   �  9   �      G   :  7   �  '   �  �  �  9   y  ?   �  6   �     *  �   C  '   ,  3   T  �   �  ;   \	  0   �	  /   �	   Append command arguments for specified CMD. These arguments should not start with '-' File that should be excluded when events triggerd Monitor file changes, and execute prepared commands. Print version and exit. Read from config file, only file name is needed, for example, if you type '--config test', then this script will find a file named test.json which locates under {}. Note: The contents in the file will override other options given as command arguments. Recusively monitor the diretory and sub directories Save the cofiguration into a file, the default file is {} Specify cmd to execute after file events triggered. If the command have its options,
            for example 'python -V', use quote " to wrap the whole command. It's recommend on parsing arguments in this way, so you can use {{name}}
            to identify which file has been changed Specify the file suffix that needs to monitor, .py extension by default Start the process or command when this scripts running. The directory to monitor, . by default. Project-Id-Version: PROJECT VERSION
Report-Msgid-Bugs-To: EMAIL@ADDRESS
POT-Creation-Date: 2019-03-10 11:03+0800
PO-Revision-Date: 2019-03-08 21:02+0800
Last-Translator: FULL NAME <EMAIL@ADDRESS>
Language: zh_Hans_CN
Language-Team: zh_Hans_CN <LL@li.org>
Plural-Forms: nplurals=1; plural=0
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit
Generated-By: Babel 2.6.0
 为指定的命令添加参数，参数不能以 - 开头 排除的文件名，当这些文件改动时不会触发事件 检查文件变动，变动后执行预设的命令。 输出版本并退出。 从配置文件中读取，只需要指定文件名，比如当你输入 '--config test'，该脚本就会在目录 {} 下面寻找叫做 test.json 的文件。注意：文件中的内容会覆盖掉命令行中其他参数的值。 递归监视文件夹和子文件夹。 将配置保存到文件中，默认的文件为 {} 指定在文件事件触发后要执行的命令。如果命令带有选项，比如运行的命令是 'python -V'，用双引号将该指令包装起来, 现在可以用 {{name}} 来获取被修改的文件名" 指定要监视的文件后缀名，默认监视 .py 文件 首次启动该脚本时就执行一次命令。 要监控的目录，默认是当前目录 '.'  