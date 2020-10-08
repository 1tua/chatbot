import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from tensorflow.keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
import string
import tkinter
from tkinter import PhotoImage
from tkinter import *
import time
import threading
import tkinter.messagebox
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import tempfile, base64, zlib
import logging
import datetime
import os


intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return np.array(bag)

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def getResponse(sentence, model, intents_json):  
    threshold=0.6
    results = model.predict(np.array([bag_of_words(sentence, words)]))[0]
    # sort by strength of probability
    results_index = np.argmax(results)
    tag = classes[results_index]
    # If probability below threshold ask users to try another question
    if results[results_index] > threshold:
        for tg in intents_json["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
                result= random.choice(responses)
        return  result
    else:
        result2= string.capwords("I didn't quite get that, try again or ask another question.")
        logger.info('Unsupported question: %s' % sentence)
        return  result2

def chatbot_response(text):
    res = getResponse(text, model, intents)
    return res
def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    if msg != '':
            ChatLog.config(state=NORMAL)
            ChatLog.insert(END, "You: " + msg + '\n\n')
            ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
            res = chatbot_response(msg)
            ChatLog.insert(END, "Bot: " + res + '\n\n')
            ChatLog.config(state=DISABLED)
            ChatLog.yview(END)

saved_username = ["You"]
window_size="600x300"

class ChatInterface(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master

        # sets default bg for top level windows
        self.tl_bg = "#EEEEEE"
        self.tl_bg2 = "#EEEEEE"
        self.tl_fg = "#000000"
        self.font = "Verdana 10"

        menu = Menu(self.master)
        self.master.config(menu=menu, bd=5)

# Menu bar

    # File
        file = Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=file)
        file.add_command(label="Clear Chat", command=self.clear_chat)
        file.add_command(label="Exit",command=self.chatexit)
        
    # Options
        options = Menu(menu, tearoff=0)
        menu.add_cascade(label="Options", menu=options)       

        
        # font
        font = Menu(options, tearoff=0)
        options.add_cascade(label="Font", menu=font)
        font.add_command(label="Default",command=self.font_change_default)
        font.add_command(label="Times",command=self.font_change_times)
        font.add_command(label="System",command=self.font_change_system)
        font.add_command(label="Helvetica",command=self.font_change_helvetica)
        font.add_command(label="Fixedsys",command=self.font_change_fixedsys)

        # color theme
        color_theme = Menu(options, tearoff=0)
        options.add_cascade(label="Color Theme", menu=color_theme)
        color_theme.add_command(label="Default",command=self.color_theme_default) 
        color_theme.add_command(label="Grey",command=self.color_theme_grey) 
        color_theme.add_command(label="Blue",command=self.color_theme_dark_blue) 
        color_theme.add_command(label="Torque",command=self.color_theme_turquoise)


      
        help_option = Menu(menu, tearoff=0)
        menu.add_cascade(label="Help", menu=help_option)
        help_option.add_command(label="About Squirro Bot", command=self.msg)
        

        self.text_frame = Frame(self.master, bd=6)
        self.text_frame.pack(expand=True, fill=BOTH)

        # scrollbar for text box
        self.text_box_scrollbar = Scrollbar(self.text_frame, bd=0)
        self.text_box_scrollbar.pack(fill=Y, side=RIGHT)

        # contains messages
        self.text_box = Text(self.text_frame, yscrollcommand=self.text_box_scrollbar.set, state=DISABLED,
                             bd=2, padx=6, pady=6, spacing3=8, wrap=WORD, bg=None, font="Verdana 10", relief=GROOVE,
                             width=10, height=1)
        self.text_box.pack(expand=True, fill=BOTH)
        self.text_box_scrollbar.config(command=self.text_box.yview)

        # frame containing user entry field
        self.entry_frame = Frame(self.master, bd=1)
        self.entry_frame.pack(side=LEFT, fill=BOTH, expand=True)

        # entry field
        self.entry_field = Entry(self.entry_frame, bd=2, justify=LEFT)
        self.entry_field.pack(fill=X, padx=6, pady=6, ipady=6)
        

        # frame containing send button and emoji button
        self.send_button_frame = Frame(self.master, bd=0)
        self.send_button_frame.pack(fill=BOTH)

        # send button
        
        root.geometry("960x600")      
        snd_button = PhotoImage(file="paper-plane.png")
        root.title("Squirro Bot")    
        root.iconbitmap('favicon.ico')

        self.send_button = Button(self.send_button_frame, image=snd_button, borderwidth=0,
                                   command=lambda: self.send_message_insert(None))
        self.send_button.pack(side=RIGHT, ipady=8)
        self.master.bind("<Return>", self.send_message_insert)
        
        self.last_sent_label(date="No messages sent.")
        root.mainloop()
   
    def last_sent_label(self, date):

        try:
            self.sent_label.destroy()
        except AttributeError:
            pass

        self.sent_label = Label(self.entry_frame, font="Verdana 7", text=date, bg=self.tl_bg2, fg=self.tl_fg)
        self.sent_label.pack(side=LEFT, fill=X, padx=3)

    def clear_chat(self):
        self.text_box.config(state=NORMAL)
        self.last_sent_label(date="No messages sent.")
        self.text_box.delete(1.0, END)
        self.text_box.delete(1.0, END)
        self.text_box.config(state=DISABLED)

    def chatexit(self):
        exit()
    
    def msg(self):
        tkinter.messagebox.showinfo("Squirro Bot v1.0",'Squirro Bot is a chatbot for answering Squirro queries')
         

    def send_message_insert(self, message):
        user_input = self.entry_field.get()
        pr1 = "Me : " + user_input + "\n"
        self.text_box.configure(state=NORMAL)
        self.text_box.insert(END, pr1)
        self.text_box.configure(state=DISABLED)
        self.text_box.see(END)
        msg = "Squirro Bot : "
        ob=chatbot_response(user_input)
        pr=msg + ob + "\n"
        self.text_box.configure(state=NORMAL)
        self.text_box.insert(END, pr)
        self.text_box.configure(state=DISABLED)
        self.text_box.see(END)
        self.last_sent_label(str(time.strftime( "Last message sent: " + '%B %d, %Y' + ' at ' + '%I:%M %p')))
        self.entry_field.delete(0,END)
        time.sleep(0)

    
    def font_change_default(self):
        self.text_box.config(font="Verdana 10")
        self.entry_field.config(font="Verdana 10")
        self.font = "Verdana 10"

    def font_change_times(self):
        self.text_box.config(font="Times")
        self.entry_field.config(font="Times")
        self.font = "Times"

    def font_change_system(self):
        self.text_box.config(font="System")
        self.entry_field.config(font="System")
        self.font = "System"

    def font_change_helvetica(self):
        self.text_box.config(font="helvetica 10")
        self.entry_field.config(font="helvetica 10")
        self.font = "helvetica 10"

    def font_change_fixedsys(self):
        self.text_box.config(font="fixedsys")
        self.entry_field.config(font="fixedsys")
        self.font = "fixedsys"

    def color_theme_default(self):
        self.master.config(bg="#EEEEEE")
        self.text_frame.config(bg="#EEEEEE")
        self.entry_frame.config(bg="#EEEEEE")
        self.text_box.config(bg="#FFFFFF", fg="#000000")
        self.entry_field.config(bg="#FFFFFF", fg="#000000", insertbackground="#000000")
        self.send_button_frame.config(bg="#EEEEEE")
        self.send_button.config(bg="#FFFFFF", fg="#000000", activebackground="#FFFFFF", activeforeground="#000000")
        self.sent_label.config(bg="#EEEEEE", fg="#000000")

        self.tl_bg = "#FFFFFF"
        self.tl_bg2 = "#EEEEEE"
        self.tl_fg = "#000000"

    # Dark
    def color_theme_dark(self):
        self.master.config(bg="#2a2b2d")
        self.text_frame.config(bg="#2a2b2d")
        self.text_box.config(bg="#212121", fg="#FFFFFF")
        self.entry_frame.config(bg="#2a2b2d")
        self.entry_field.config(bg="#212121", fg="#FFFFFF", insertbackground="#FFFFFF")
        self.send_button_frame.config(bg="#2a2b2d")
        self.send_button.config(bg="#212121", fg="#FFFFFF", activebackground="#212121", activeforeground="#FFFFFF")
        self.sent_label.config(bg="#2a2b2d", fg="#FFFFFF")

        self.tl_bg = "#212121"
        self.tl_bg2 = "#2a2b2d"
        self.tl_fg = "#FFFFFF"

    # Grey
    def color_theme_grey(self):
        self.master.config(bg="#444444")
        self.text_frame.config(bg="#444444")
        self.text_box.config(bg="#4f4f4f", fg="#ffffff")
        self.entry_frame.config(bg="#444444")
        self.entry_field.config(bg="#4f4f4f", fg="#ffffff", insertbackground="#ffffff")
        self.send_button_frame.config(bg="#444444")
        self.send_button.config(bg="#4f4f4f", fg="#ffffff", activebackground="#4f4f4f", activeforeground="#ffffff")
        self.sent_label.config(bg="#444444", fg="#ffffff")

        self.tl_bg = "#4f4f4f"
        self.tl_bg2 = "#444444"
        self.tl_fg = "#ffffff"


    def color_theme_turquoise(self):
        self.master.config(bg="#003333")
        self.text_frame.config(bg="#003333")
        self.text_box.config(bg="#669999", fg="#FFFFFF")
        self.entry_frame.config(bg="#003333")
        self.entry_field.config(bg="#669999", fg="#FFFFFF", insertbackground="#FFFFFF")
        self.send_button_frame.config(bg="#003333")
        self.send_button.config(bg="#669999", fg="#FFFFFF", activebackground="#669999", activeforeground="#FFFFFF")
        self.sent_label.config(bg="#003333", fg="#FFFFFF")

        self.tl_bg = "#669999"
        self.tl_bg2 = "#003333"
        self.tl_fg = "#FFFFFF"    

    # Blue
    def color_theme_dark_blue(self):
        self.master.config(bg="#263b54")
        self.text_frame.config(bg="#263b54")
        self.text_box.config(bg="#1c2e44", fg="#FFFFFF")
        self.entry_frame.config(bg="#263b54")
        self.entry_field.config(bg="#1c2e44", fg="#FFFFFF", insertbackground="#FFFFFF")
        self.send_button_frame.config(bg="#263b54")
        self.send_button.config(bg="#1c2e44", fg="#FFFFFF", activebackground="#1c2e44", activeforeground="#FFFFFF")
        self.sent_label.config(bg="#263b54", fg="#FFFFFF")

        self.tl_bg = "#1c2e44"
        self.tl_bg2 = "#263b54"
        self.tl_fg = "#FFFFFF"

 
    

    # Torque
    def color_theme_turquoise(self):
        self.master.config(bg="#003333")
        self.text_frame.config(bg="#003333")
        self.text_box.config(bg="#669999", fg="#FFFFFF")
        self.entry_frame.config(bg="#003333")
        self.entry_field.config(bg="#669999", fg="#FFFFFF", insertbackground="#FFFFFF")
        self.send_button_frame.config(bg="#003333")
        self.send_button.config(bg="#669999", fg="#FFFFFF", activebackground="#669999", activeforeground="#FFFFFF")
        self.sent_label.config(bg="#003333", fg="#FFFFFF")

        self.tl_bg = "#669999"
        self.tl_bg2 = "#003333"
        self.tl_fg = "#FFFFFF"

   

    # Default font and color theme
    def default_format(self):
        self.font_change_default()
        self.color_theme_default()    
   
   
# Create application logger
logger_name = 'app_logger'
logger_level = logging.DEBUG
logger = logging.getLogger(logger_name)
logger.setLevel(logger_level)

# Create file handler
log_file_name = 'chatbot_logs.log'
log_folder_path = '//sd1filepw11/apps/W Drive New Structure/Regulatory Operations/Chief Operating Office/Innovation/SquirroBotLogs'
log_file_path = os.path.join(log_folder_path, log_file_name)
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logger_level)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logger_level)

# Format log messages
log_message_format = '[%(asctime)s] [%(levelname)s] [%(filename)s] [Line %(lineno)s] %(message)s'
log_message_date_format = '%Y-%m-%d %H:%M:%S'
log_formatter = logging.Formatter(log_message_format, log_message_date_format)
file_handler.setFormatter(log_formatter)
console_handler.setFormatter(log_formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

        
root=Tk()


a = ChatInterface(root)
root.geometry(window_size)
root.mainloop()


