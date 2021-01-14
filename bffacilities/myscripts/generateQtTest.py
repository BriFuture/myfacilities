import jinja2
from jinja2 import Template
from bffacilities._constants import BFF_TEMPLATE_PATH
import os
import os.path as osp
import tqdm

tempateLoader = jinja2.FileSystemLoader(searchpath = str(BFF_TEMPLATE_PATH))
templateEnv = jinja2.Environment(loader=tempateLoader)
templateFile = "generateQtTest.jinja2"
testFileTemp = templateEnv.get_template(templateFile)
cmakeTemp = templateEnv.get_template("generateQtTestCmake.jinja2")


def writeIntoTestFile(args):
    """args
    {
        "className":  Generator,
        "file": generator,
        "methods": [ "method1", "method2" ]
    }
    """
    className = args["className"]
    testClassName = className if className.endswith("Test") else f"{className}Test"
    args["include"] = className
    args["className"] = testClassName

    if "methods" not in args:
        args["methods"] = ["method1", "method2"]
    methods = args["methods"]
    for i in range(len(methods)):
        m = methods[i]
        methods[i] = f"{m[0].upper()}{m[1:]}"
    content = testFileTemp.render(args)
    fileName = args.get("file")
    if fileName is None:
        print(content)
    else:
        filePath = osp.split(fileName)
        root = filePath[0]
        if root is None:
            root = "."
        fileName = filePath[1]
        if not fileName.startswith("tst_"):
            fileName = f"tst_{fileName.lower()}"
        if fileName.endswith("test"):
            fileName = f"{fileName}.cpp"
        if not fileName.endswith("test.cpp"):
            fileName = f"{fileName.lower()}test.cpp"
        with open(osp.join(root, fileName), "w") as f:
            f.write(content)
import clang.cindex
from clang.cindex import CursorKind, AccessSpecifier, TranslationUnit
import asciitree

class CxxParser():
    def __init__(self):
        self._methods = {}  # key is className, value is array which contained all public methods
        self.index = clang.cindex.Index.create()
        self.cur_file = None
        self.cxx_args = ["-x", "c++"]

    def parse(self, file):
        filePath = osp.split(file)
        root = filePath[0]
        self.cur_file = filePath[1]
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            content = content.replace("signals", "private")
            content = content.replace("Q_OBJECT", "")
            content = content.replace("slots", "")
            tu = self.index.parse(self.cur_file, self.cxx_args,
                [(self.cur_file, content)])
        assert isinstance(tu, TranslationUnit)
        self.iterateNode(tu.cursor)
        # print(asciitree.draw_tree(tu.cursor, self.node_children, self.process_node))

    def _iterate(self, parent, method):
        for c in parent.get_children() :
            if c.location.file.name != self.cur_file:
                continue
            method(c)

    def iterateNode(self, node):
        p = self.processNode(node)
        if p:
            self._iterate(node, self.iterateNode)
    
    def processNode(self, node):
        kind = node.kind
        # text = node.spelling if len(node.spelling) > 0 else node.displayname
        if kind == CursorKind.CLASS_DECL:
            cn = node.spelling
            # print("\t class name ********* ", cn)
            if cn not in self._methods:
                self._methods[cn] = set()
            self._lastMethod = cn
            self._methodPublic = False
            self._iterate(node, self.processClass)
            return False
        else:
            # print(f'{kind} ==> {text} ')
            return True

    def processClass(self, node):
        kind = node.kind
        if kind == CursorKind.CXX_ACCESS_SPEC_DECL:
            declName = node.access_specifier.name
            if declName == "PUBLIC":
                self._methodPublic = True
                # print(f"\t{declName}")
            else:
                self._methodPublic = False

        elif kind == CursorKind.CXX_METHOD:
            if self._methodPublic:
                self._methods[self._lastMethod].add(node.spelling)
                # print(f' public method ==> {node.spelling} ')
            else:
                pass
                # print(f' private method ==> {node.spelling} ')
        else:
            # print(f' class {kind} ==> {node.spelling} ')
            pass
    @property
    def methods(self):
        remove = []
        for cn, ms in self._methods.items():
            if len(ms) == 0:
                remove.append(cn)
        return { k: v for k, v in self._methods.items() if k not in remove }
    def node_children(self, node):
        return list(c for c in node.get_children() 
            if c.location.file.name == self.cur_file
            )
    def process_node(self, node):
        text = node.spelling if len(node.spelling) > 0 else node.displayname
        kind = node.kind
        if kind == CursorKind.CLASS_DECL:
            if kind.name not in self._methods:
                self._methods[kind.name] = []
        if kind == CursorKind.CXX_ACCESS_SPEC_DECL:
            return f"{node.access_specifier.name}"
        return f'{kind} ==> {text} '

def saveClassTests(methods, project = None, file=None):
    # print(methods)
    print(f"Writing {len(methods)} items")

    for cn, ms in methods.items():
        args = { "className": cn, "methods": list(ms), 
            "file": cn.lower() 
            }
        if file is not None:
            args["file"] = file
        # print(args)
        writeIntoTestFile(args)
    if project is None:
        content = cmakeTemp.render(project="Unnamed", classes=methods.keys())
        print(content)
    else:
        content = cmakeTemp.render(project=project, classes=methods.keys())
        with open("Temp_CMakeLists.txt", "w") as f:
            f.write(content)
    # pass

def testClang(args):
    cparser = CxxParser()

    if args["input"]:
        file = args["input"]
        cparser.parse(file)
        # print(file, cparser.methods)
        saveClassTests(cparser.methods, project=args["project"], file=args["output"])
        return True
    elif args["dir"]:
        d = args["dir"]
        # print("============")
        exclude = args.get("exclude_dir", "")
        if len(exclude) > 0:
            exclude = exclude.split(",")
        else:
            exclude = []
        exclude.append(".git")
        exclude.append("build")
        def isExcludeDir(root):
            for e in exclude:
                if e in root:
                    return True
            return False

        def isExcludeFile(file):
            if f.endswith("_p.h"):
                return True
            if f.endswith("pch.h"):
                return True
            return False
        for root, _, files in os.walk(d):
            # for d in dirs:
            #     print(root," Dir --- ", d)
            if isExcludeDir(root): 
                continue
            for f in tqdm.tqdm(files, desc=f"Searching for {root}"):
                # print(root, exclude)
                if isExcludeFile(f): continue

                if f.endswith(".h"):
                    # print("file", root, f)
                    cparser.parse(osp.join(root, f))
        saveClassTests(cparser.methods, project=args["project"])
        
        return True
    return False

def main(args = None):
    from argparse import ArgumentParser
    parser = ArgumentParser(prog="generateQtTest", description="Generate Qt Test File mannually")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-d", "--dir", type=str, help="The directory that program will walk through")
    group.add_argument("-f", "--input", type=str, help="Input content from file")
    group.add_argument("-c", "--className", type=str, help="Class Name That will be used in Test")

    parser.add_argument("-p", "--project", type=str, help="Project name if you want to write test content into CMakeLists.txt")
    parser.add_argument("-o", "--output", type=str, help="Output into file")
    parser.add_argument("-e", "--exclude-dir", type=str, help="Exclude directory, very useful when using directory `-d`")

    args = vars(parser.parse_args(args))
    if args["className"]:
        args["file"] = args["output"]
        writeIntoTestFile(args)
    else:
        if not testClang(args):
            parser.print_help()
