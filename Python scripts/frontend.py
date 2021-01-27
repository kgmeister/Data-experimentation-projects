import backend
import wx

def press_button(event):
    meters = float(input_box.GetValue())
    result.SetLabel(str(backend.meters_feet(meters)))

app = wx.App()
frame = wx.Frame(parent = None, title = "Meters to Feet")

panel = wx.Panel(frame)
sizer = wx.GridBagSizer()

input_label = wx.StaticText(panel, label = "Meters: ")
input_box = wx.TextCtrl(panel)
result_label = wx.StaticText(panel, label = "Feet: ")
result = wx.StaticText(panel, label = "")
button = wx.Button(panel, label = "Convert")
button.Bind(wx.EVT_BUTTON, press_button)


sizer.Add(input_label, (0,0))
sizer.Add(input_box, (0,1))
sizer.Add(result_label, (1,0))
sizer.Add(result, (1,1))
sizer.Add(button, (2,0))


panel.SetSizerAndFit(sizer)
frame.SetSizerAndFit(sizer)
frame.Show()
app.MainLoop()