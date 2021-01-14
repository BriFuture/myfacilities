if (WScript.Arguments.length < 1) {
    msg = "No message supplied"
    a = {"te": 13, "b": null}
    WScript.Echo(a.b)
    WScript.Quit(1);
}
function test(arg) {
    WScript.Echo(arg)
}
var msg = WScript.Arguments.Item(0);
if(msg.length > 0 ) {
    test(msg + "\\test")
}
// for (i = 0; i < WScript.Arguments.length; i++) {
    // msg = msg + i + WScript.Arguments.Item(i) + " ";
// }
var WshShell = WScript.CreateObject("WScript.Shell");
var Startup = WshShell.SpecialFolders("Startup");
// var x=msgbox("Your Text Here" ,0, Startup)
WScript.Echo(Startup)