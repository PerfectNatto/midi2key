import pygame.midi as m
import pyautogui
import webview
import json

pyautogui.PAUSE = 0.016

class Api:

    def __init__(self):
        self.running = False
        self.port = 0
        json_open = open('keybind.json', 'r')
        json_load = json.load(json_open)
        self.keybind_dict = { json_load[i]['midi'] : json_load[i]['key'] for i in range( len(json_load) ) }
        print(self.keybind_dict)
    
    def getMidiInfo(self):
        m.init()
        i_num = m.get_count()
        devices = []
        input = []
        output = []
        for i in range(i_num):
            print(m.get_device_info(i))
            if m.get_device_info(i)[2] == 1 :
                devices.append([m.get_device_info(i)[1].decode('utf8'),i])
        print(devices)
        return devices
    
    def startDevice(self, inputNum):
        self.port = m.Input(inputNum)
        self.running = True
        while self.running:
            if self.port.poll():
                midi_events = self.port.read(4)
                print(f'events: {midi_events[0][0][0]}')
                if midi_events[0][0][0] == 144:
                    key = self.keybind_dict.get(midi_events[0][0][1])
                    if key:
                        pyautogui.keyDown(key)
                if midi_events[0][0][0] == 128:
                    key = self.keybind_dict.get(midi_events[0][0][1])
                    if key:
                        pyautogui.keyUp(key)
    
    def stopDevice(self):
        self.running = False
        self.port.close()

    def get_keybind_dict(self):
       return json.dumps(self.keybind_dict)

api=Api()
window = webview.create_window("midi2key", url="index.html", js_api=api)
webview.start(gui="cef",debug=True,http_server=False)