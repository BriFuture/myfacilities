/**
 * @params 
 * -w <working dir> (necessary)
 * -tp <target path>, target to specified .exe  (necessary)
 * -link <Link File Location>  (necessary)
 * -desc <description>
 * -d <desktop dir>, copy to desktop
 * -s <startup dir>, copy to startup
 */
if (WScript.Arguments.length < 1) {
    WScript.Quit(1);
}
var args = {    "workingdir": null,
    "target": null,
    "link": false,
    "desc": "bff program link",
    "desktop": false,
    "startup": false
}

for (var i = 0; i < WScript.Arguments.length; i++) {
    var arg = WScript.Arguments.Item(i);
    if(arg === "--workingdir") {
        i ++;
        a = WScript.Arguments.Item(i);
        args.workingdir = a;
    } else if(arg === "--target") {
        i++;
        a = WScript.Arguments.Item(i);
        args.target = a;
    } else if(arg === "--link") {
        i++;
        a = WScript.Arguments.Item(i);
        args.link = a;
    } else if(arg === "--desc") {
        i++;
        a = WScript.Arguments.Item(i);
        args.desc = a;
    } else if(arg === "--desktop") {
        i++; // similar with link
        a = WScript.Arguments.Item(i);
        args.desktop = a;
    } else if(arg === "--startup") {
        i++; // similar with link
        a = WScript.Arguments.Item(i);
        args.startup = a;
    }
}

if(args.workingdir.length === 0) {
    WScript.Quit(1)
}
if(args.target.length === 0) {
    WScript.Quit(1)
}
linklen = args.link || args.desktop || args.startup;
if(!linklen) {
    WScript.Quit(1)
}

function create(workingdir, link, target, desc) {
    var oShellLink = WshShell.CreateShortcut(link)
    oShellLink.TargetPath = target;
    oShellLink.WindowStyle = 1;
    oShellLink.Description = desc;
    
    oShellLink.WorkingDirectory = workingdir;
    oShellLink.Save();
}

var WshShell = WScript.CreateObject("WScript.Shell");
if(args.desktop.length > 0) {
    var link = WshShell.SpecialFolders("Desktop") + "\\" + args.desktop + ".lnk";
    create(args.workingdir, link, args.target, args.desc);
    // WScript.echo("Create Desktop");
}
if(args.startup.length > 0) {
    var link = WshShell.SpecialFolders("Startup") + "\\" + args.startup + ".lnk";
    create(args.workingdir, link, args.target, args.desc);
    // WScript.echo("Create Startup");
}
if(args.link.length > 0) {
    create(args.workingdir, args.link, args.target, args.desc);
    // WScript.echo("Create Link" + args.link);
}