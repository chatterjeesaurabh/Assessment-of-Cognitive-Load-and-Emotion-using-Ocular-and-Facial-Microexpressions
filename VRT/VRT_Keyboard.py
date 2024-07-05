import pygame
import os
import random
import string
import time
import csv
import numpy as np
from pygame.locals import *
from win32api import GetSystemMetrics
#from PIL import Image, ImageTk
import sys
#from tkinter.ttk import *
#from tkinter import ttk, filedialog


LARGE_FONT= ("Times", 20)
MEDIUM_FONT= ("Times", 16)




pygame.quit()
pygame.init()
white=(255,255,255)
black=(0,0,0)
red=(255,0,0)
purple=(238,130,238)
blue=(0,0,255)
dark_red=(177,76,34)
light_green=(0,255,0)
green=(34,177,76)
global oo
global j
global mean_time
#w = GetSystemMetrics(0) 
#h = GetSystemMetrics(1)
w = 1200
h = 700


j=0
gameDisplay = pygame.display.set_mode((1200, 700),pygame.RESIZABLE)
pygame.display.set_caption("GAME")
clock=pygame.time.Clock()
font=pygame.font.SysFont('Monotype Corsiva',32)
smallfont=pygame.font.SysFont('Arial',12)
medfont=pygame.font.SysFont('Arial',34)
largefont=pygame.font.SysFont('Arial Black',48)
gameDisplay.fill(white)
def get_red(s_3):
   if s_3 == 1:
     gameDisplay.blit(image_2, (w/2,0))

   elif s_3 == 2:
     gameDisplay.blit(image_2, (0,h/2))
     
   elif s_3 == 3:
     gameDisplay.blit(image_2, (w/2,h))
              
   elif s_3 == 4:
     gameDisplay.blit(image_2, (w,h/2))
   


def get_key():
   while 1:
     event = pygame.event.poll()
     if event.type == KEYDOWN:
       return event.key
     else:
       pass
global image_library
image_library = {}
def get_image(path):
      
      image = image_library.get(path)
      if image == None:
              canonicalized_path = path.replace('/', os.sep).replace('\\', os.sep)
              image = pygame.image.load(canonicalized_path)
              image_library[path] = image
      return image

s1="No of correct responses are:"
s2=" No of incorrect responses are:"
s3="Mean time for correct response is:"
s4="seconds"
image_1 = get_image('green _ball.gif')
image_2=get_image('red_ball.gif')
#image_3=get_image('both.gif')
#image_4=get_image('gdown.gif')
#image_5=get_image('gleft.gif')
#image_6=get_image('gright.gif')
#image_7=get_image('gtop.gif')
#image_8=get_image('level2.gif')
#image_3=get_image("frontview8.gif")
#image_4=get_image("capture1.gif")
#image_5=get_image("Decision1.gif")
#image_6=get_image("start.gif")

def gameLoop():
  #print("gameLoop")
  joysticks = [] 
  gameExit = False
  gameOver = False
  gameOver_1=False
  crct=0
  incrt=0
  keepplaying=True
  c=0
  t1=0
  t2=0
  t3=0
  t4=0
  t=0
  b=0
  global oo
  oo=0
  gameDisplay.fill(white)
  pygame.display.update()

  joysticks = []
  clock = pygame.time.Clock()
  keepPlaying = True

  for i in range(0, pygame.joystick.get_count()):
        joysticks.append(pygame.joystick.Joystick(i))
        joysticks[-1].init()

  
  screen_text_0= font.render("                                                    Welcome to the Psychomotor Vigilance Test  Level 1                                          ", True,red)
 
  gameDisplay.blit(screen_text_0, [0,300])
  pygame.display.update()
  time.sleep(2)

  """image_gl=get_image('gleft.gif')
  gameDisplay.blit(image_gl, (0,0))
  time.sleep(5)
  pygame.display.update()
  gameDisplay.fill(white)"""



  gameDisplay.fill(white)
      
  screen_text = font.render("                                       Press the right-handed colored buttons as per the Direction of the Green Ball !!!. ", True,red)
  gameDisplay.blit(screen_text, [50,300])
  pygame.display.update()
  file = 'game1.wav'
  pygame.mixer.init()
  pygame.mixer.music.load(file)
  pygame.mixer.music.play()
  pygame.event.wait()
  time.sleep(1)
  gameDisplay.fill(white)
  screen_text_1= font.render("                                             Game Starts!!! Wish you all the LUCK!!!", True,red)
  gameDisplay.blit(screen_text_1, [0,300])
  pygame.display.update()
  time.sleep(1)
  gameDisplay.fill(white)

  for counter in range(1,11):
  #joysticks[-1].init()
      t1=t2=t3=t4=0
      s=random.randint(1, 4)
      if s == 1:
        clock.tick(60)

        gameDisplay.blit(image_1, (w/2,0))
        pygame.display.update()
        start_1=time.time()
        
        while (time.time()-start_1)<1 and keepplaying:
          clock.tick(60)
          for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit() 
                sys.exit()

            if event.type==pygame.KEYDOWN:
                end_1=time.time()
                t1=end_1-start_1
                m=event.key
                #print ("Joystick '",joysticks[event.joy].get_name(),"' button",m,"down.")
                if event.key==pygame.K_UP:
                   crct=crct+1
                   t2=t3=t4=0
                   #print(t1)
                keepplaying = False
        if t1 != 0:
          time.sleep(1.05-t1)
        gameDisplay.fill(white)
        pygame.display.update()
        time.sleep(0.1)     
        keepplaying = True

      if s == 2:
        clock.tick(60)
        gameDisplay.blit(image_1, (0,h/2))
        pygame.display.update()
        start_1=time.time()
        
        while (time.time()-start_1)<1 and keepplaying:
          clock.tick(60)
          for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit() 
                sys.exit()

            if event.type==pygame.KEYDOWN:
                end_1=time.time()
                t2=end_1-start_1
                m=event.key
                #print ("Joystick '",joysticks[event.joy].get_name(),"' button",m,"down.")
                if event.key==pygame.K_LEFT:
                   crct=crct+1
                   t1=t3=t4=0
                   #print(t2)
                keepplaying = False
        if t2 != 0:
          time.sleep(1.05-t2)
        gameDisplay.fill(white)
        pygame.display.update()
        time.sleep(0.1)     
        keepplaying = True

      if s == 3:
        clock.tick(60)
        gameDisplay.blit(image_1, (w/2,h-100))
        pygame.display.update()
        start_1=time.time()
        
        while (time.time()-start_1)<1 and keepplaying:
          clock.tick(60)
          for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit() 
                sys.exit()

            if event.type==pygame.KEYDOWN:
                end_1=time.time()
                t3=end_1-start_1
                m=event.key
                #print ("Joystick '",joysticks[event.joy].get_name(),"' button",m,"down.")
                if event.key==pygame.K_DOWN:
                   crct=crct+1
                   t2=t1=t4=0
                   #print(t3)
                keepplaying = False
        if t3 != 0:
          time.sleep(1.05-t3)
        gameDisplay.fill(white)
        pygame.display.update()
        time.sleep(0.1)     
        keepplaying = True

      if s == 4:
        clock.tick(60)
        gameDisplay.blit(image_1, (w-100,h/2))
        pygame.display.update()
        start_1=time.time()
        
        while (time.time()-start_1)<1 and keepplaying:
          clock.tick(60)
          for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit() 
                sys.exit()

            if event.type==pygame.KEYDOWN:
                end_1=time.time()
                t4=end_1-start_1
                m=event.key
                #print ("Joystick '",joysticks[event.joy].get_name(),"' button",m,"down.")
                if event.key==pygame.K_RIGHT:
                   crct=crct+1
                   t2=t3=t1=0
                   #print(t4)
                keepplaying = False
        if t4 != 0:
          time.sleep(1.05-t4)
        gameDisplay.fill(white)
        pygame.display.update()
        time.sleep(0.1)     
        keepplaying = True
               
      else: incrt=incrt+1
      t=t+(t1+t2+t3+t4)
  #print("time",t)          
  incorrect=10-crct
  global mean_time
  if crct == 0:
      mean_time = 0
  else:
      mean_time=(t/crct)
  gameDisplay.fill(white)
  screen_text = font.render(str(s1+str(crct)), True,black)
  gameDisplay.blit(screen_text, [550,100])
  pygame.display.update()
  screen_text = font.render(str(s2+str(incorrect)), True,black)
  gameDisplay.blit(screen_text, [550,200])
  pygame.display.update()
  screen_text = font.render(str(s3+str(mean_time)+s4), True,blue)
  gameDisplay.blit(screen_text, [550,300])
  pygame.display.update()
  screen_text = font.render("Advancing to Level 2 in 2 seconds", True,purple)
  gameDisplay.blit(screen_text, [550,400])
  pygame.display.update()

  print(f"Level 1: \n Correct: {crct} \n Incorrect: {incorrect} \n Mean Time: {mean_time} \n \n")

  time.sleep(2)
  gameDisplay.fill(white)
  pygame.display.update()

  ####################################################################################################################################
  crct_1=0
  incrt_1=0
  incorrect_1=0
  c_1=0
  t1_1=0
  t2_1=0
  t3_1=0
  t4_1=0
  t_1=0
  total_crct_1=0
  screen_text_0= font.render("                                                    Welcome to the Psychomotor Vigilance Test Level 2                                          ", True,red)
 
  gameDisplay.blit(screen_text_0, [0,300])
  pygame.display.update()
  #time.sleep(5)
  
  #gameDisplay.blit(image_8, (0,0))
 # pygame.time.delay(5000)
  pygame.display.update()
  gameDisplay.fill(white)
      
 # gameDisplay.blit(image_8, (0,0))
 # pygame.time.delay(5000)
  pygame.display.update()
  gameDisplay.fill(white)
  screen_text = font.render("                                                   Press for a Green Ball but not for a Red Ball      ", True,blue)
  gameDisplay.blit(screen_text, [50,300])
  pygame.display.update()
  file = 'Game2.mp3'
  pygame.mixer.init()
  pygame.mixer.music.load(file)
  pygame.mixer.music.play()
  pygame.event.wait()
  time.sleep(1)
  gameDisplay.fill(white)
  screen_text_1= font.render("                                                                 Game Starts!!! Wish you all the LUCK!!!"                                                                 , True,blue)
  gameDisplay.blit(screen_text_1, [0,300])
  pygame.display.update()
  time.sleep(1)
  gameDisplay.fill(white)

  for counter in range(1,11):
      s=random.randint(1, 4)
      s_1 = random.randint(1,2)
      t1_1=t2_1=t3_1=t4_1=0
      s#_2=random.randint(
      if s == 1:
            clock.tick(60)
            if s_1 == 1:
                gameDisplay.blit(image_1, (w/2,0))
                total_crct_1 = total_crct_1 + 1
            elif s_1 == 2:
              gameDisplay.blit(image_2, (w/2,0))
            pygame.display.update()
            start_1_1= time.time()
            while (time.time()-start_1_1)<1 and keepplaying:
              clock.tick(60)
              for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit() 
                    sys.exit()

                if event.type==pygame.KEYDOWN:
                    end_1_1=time.time()
                    t1_1=end_1_1-start_1_1
                    m=event.key
                    #print ("Joystick '",joysticks[event.joy].get_name(),"' button",m,"down.")
                    if event.key==pygame.K_UP and s_1==1:
                       crct_1=crct_1+1
                       t2_1=t3_1=t4_1=0
                       #print("t1_1",t1_1)
                    else:
                        incrt_1 = incrt_1 + 1
                    keepplaying = False
            if t1_1 != 0:
              time.sleep(1.05-t1_1)
            gameDisplay.fill(white)
            pygame.display.update()
            time.sleep(0.1)     
            keepplaying = True
                       
             
            
      elif s == 2:
              clock.tick(60)
              if s_1 == 1:
                    gameDisplay.blit(image_1, (0,h/2))
                    total_crct_1 = total_crct_1 +1
              elif s_1==2:
                  gameDisplay.blit(image_2, (0,h/2))
              pygame.display.update()
              start_2_1= time.time()
              while (time.time()-start_2_1)<1 and keepplaying:
                clock.tick(60)
                for event in pygame.event.get():
                  if event.type == pygame.QUIT:
                    pygame.quit() 
                    sys.exit()

                  if event.type==pygame.KEYDOWN:
                      end_2_1=time.time()
                      t2_1=end_2_1-start_2_1
                      m=event.key
                      #print ("Joystick '",joysticks[event.joy].get_name(),"' button",m,"down.")
                      if event.key==pygame.K_LEFT and s_1==1:
                         crct_1=crct_1+1
                         t1_1=t3_1=t4_1=0
                         #print("t2_1",t2_1)
                      else:
                        incrt_1 = incrt_1 + 1
                      keepplaying = False
              if t2_1 != 0:
                time.sleep(1.05-t2_1)
              gameDisplay.fill(white)
              pygame.display.update()
              time.sleep(0.1)     
              keepplaying = True
                           
       
              
      elif s == 3:
             clock.tick(60)
             if s_1==1:
               gameDisplay.blit(image_1,(w/2,h-100))
               total_crct_1 = total_crct_1 + 1
             elif s_1==2:
               gameDisplay.blit(image_2,(w/2,h-100))
             pygame.display.update()
             start_3_1= time.time()
             while (time.time()-start_3_1)<1 and keepplaying:
                clock.tick(60)
                for event in pygame.event.get():
                  if event.type == pygame.QUIT:
                      pygame.quit() 
                      sys.exit()

                  if event.type==pygame.KEYDOWN:
                      end_3_1=time.time()
                      t3_1=end_3_1-start_3_1
                      m=event.key
                      #print ("Joystick '",joysticks[event.joy].get_name(),"' button",m,"down.")
                      if event.key==pygame.K_DOWN and s_1==1:
                         crct_1=crct_1+1
                         t2_1=t1_1=t4_1=0
                         #print("t3_1",t3_1)
                      else:
                        incrt_1 = incrt_1 + 1
                      keepplaying = False
             if t3_1 != 0:
                time.sleep(1.05-t3_1)
             gameDisplay.fill(white)
             pygame.display.update()
             time.sleep(0.1)     
             keepplaying = True
         
              
      elif s == 4:
             clock.tick(60)
             if s_1==1:
                  gameDisplay.blit(image_1,(w-100,h/2))
                  total_crct_1 = total_crct_1 + 1
             elif s_1==2:
                gameDisplay.blit(image_2,(w-100,h/2)) 
             pygame.display.update()
             start_4_1= time.time()
             while (time.time()-start_4_1)<1 and keepplaying:
                clock.tick(60)
                for event in pygame.event.get():
                  if event.type == pygame.QUIT:
                      pygame.quit() 
                      sys.exit()

                  if event.type==pygame.KEYDOWN:
                      end_4_1=time.time()
                      t4_1=end_4_1-start_4_1
                      m=event.key
                      #print ("Joystick '",joysticks[event.joy].get_name(),"' button",m,"down.")
                      if event.key==pygame.K_RIGHT and s_1==1:
                         crct_1=crct_1+1
                         t2_1=t3_1=t1_1=0
                         #print("t4_1",t4_1)
                      else:
                        incrt_1 = incrt_1 + 1
                      keepplaying = False
             if t4_1 != 0:
                time.sleep(1.05-t4_1)
             gameDisplay.fill(white)
             pygame.display.update()
             time.sleep(0.1)     
             keepplaying = True
         
      if s_1 == 1:        
        t_1=t_1+(t1_1+t2_1+t3_1+t4_1)
      #print("time",t_1)
  incorrect_1=total_crct_1 - crct_1+incrt_1
  if crct_1 == 0:
      mean_time_1 = 0
      crct_1=10-incorrect_1
  else:
      mean_time_1=t_1/crct_1
      crct_1=10-incorrect_1
  gameDisplay.fill(white)
  screen_text = font.render(str(s1+str(crct_1)), True,black)
  gameDisplay.blit(screen_text, [550,100])
  pygame.display.update()
  screen_text = font.render(str(s2+str(incorrect_1)), True,black)
  gameDisplay.blit(screen_text, [550,200])
  pygame.display.update()
  screen_text = font.render(str(s3+str(mean_time_1)+s4), True,blue)
  gameDisplay.blit(screen_text, [550,300])
  pygame.display.update()
  screen_text = font.render("Advancing to level_3 in 8 seconds", True,purple)
  gameDisplay.blit(screen_text, [550,400])
  pygame.display.update()

  print(f"Level 2: \n Correct: {crct_1} \n Incorrect: {incorrect_1} \n Mean Time: {mean_time_1} \n \n")

  time.sleep(2)
  gameDisplay.fill(white)
  pygame.display.update()

  ############################################################################################################################
  ####################################################################################################################################
  crct_2=0
  incrt_2=0
  incorrect_2=0
  c_2=0
  t1_2=0
  t2_2=0
  t3_2=0
  t4_2=0
  t_2=0
  total_crct_2=0
  screen_text_0= font.render("                                          Welcome to the Psychomotor Vigilance Test  Level 3                                          ", True,red)
  gameDisplay.blit(screen_text_0, [0,300])
  pygame.display.update()
  #time.sleep(5)
  #gameDisplay.blit(image_3, (0,0))
  #pygame.time.delay(5000)
  #pygame.display.update()
  #gameDisplay.fill(white)
      
  #gameDisplay.blit(image_3, (0,0))
  #pygame.time.delay(5000)
  pygame.display.update()
  gameDisplay.fill(white)


  screen_text = font.render("                                    Press the right-handed or left-handed buttons as per the color of the ball is Green or Red !!!. ", True,blue)
  gameDisplay.blit(screen_text, [50,300])
  pygame.display.update()
  file = 'game3.mp3'
  pygame.mixer.init()
  pygame.mixer.music.load(file)
  pygame.mixer.music.play()
  pygame.event.wait()
  time.sleep(1)
  gameDisplay.fill(white)
  screen_text_1= font.render("                                                                 Game Starts!!! Wish you all the LUCK!!!"                                                                 , True,blue)
  gameDisplay.blit(screen_text_1, [0,300])
  pygame.display.update()
  time.sleep(1)
  gameDisplay.fill(white)

  for counter in range(1,11):
      s=random.randint(1, 4)
      s_1 = random.randint(1, 2)
      t1_2=t2_2=t3_2=t4_2=0
      if s == 1:
            clock.tick(60)
            if s_1 == 1:
              gameDisplay.blit(image_1, (w/2,0))
              pygame.display.update()
              start_1_2= time.time()
              while (time.time()-start_1_2)<1 and keepplaying:
                clock.tick(60)
                for event in pygame.event.get():
                  if event.type == pygame.QUIT:
                      pygame.quit() 
                      sys.exit()

                  if event.type==pygame.KEYDOWN:
                      end_1_2=time.time()
                      t1_2=end_1_2-start_1_2
                      m=event.key
                      #print ("Joystick '",joysticks[event.joy].get_name(),"' button",m,"down.")
                      if (event.key==pygame.K_RIGHT):
                         crct_2=crct_2+1
                         t_2 = t_2 + t1_2
                         t2_2=t3_2=t4_2=0
                         #print("t1_2",t1_2)
                      keepplaying = False
              if t1_2 != 0:
                time.sleep(1.05-t1_2)
              gameDisplay.fill(white)
              pygame.display.update()
              time.sleep(0.1)     
              keepplaying = True

            elif s_1 == 2:
              clock.tick(60)
              gameDisplay.blit(image_2, (w/2,0))
              pygame.display.update()
              start_1_2= time.time()
              while (time.time()-start_1_2)<1 and keepplaying:
                clock.tick(60)
                for event in pygame.event.get():
                  if event.type == pygame.QUIT:
                      pygame.quit() 
                      sys.exit()

                  if event.type==pygame.KEYDOWN:
                      end_1_2=time.time()
                      t1_2=end_1_2-start_1_2
                      m=event.key
                      #print ("Joystick '",joysticks[event.joy].get_name(),"' button",m,"down.")
                      if (event.key==pygame.K_LEFT):
                        crct_2=crct_2+1
                        t_2 = t_2 + t1_2
                        t2_2=t3_2=t4_2=0
                        #print("t1_2",t1_2)
                      keepplaying = False
              if t1_2 != 0:
                time.sleep(1.05-t1_2)
              gameDisplay.fill(white)
              pygame.display.update()
              time.sleep(0.1)     
              keepplaying = True
                       
             
            
      elif s == 2:
              clock.tick(60)
              if s_1 == 1:
                  gameDisplay.blit(image_1, (0,h/2))
                  pygame.display.update()
                  start_2_2= time.time()
                  while (time.time()-start_2_2)<1 and keepplaying:
                    clock.tick(60)
                    for event in pygame.event.get():
                      if event.type == pygame.QUIT:
                          pygame.quit() 
                          sys.exit()

                      if event.type==pygame.KEYDOWN:
                          end_2_2=time.time()
                          t2_2=end_2_2-start_2_2
                          m=event.key
                          #print ("Joystick '",joysticks[event.joy].get_name(),"' button",m,"down.")
                          if (event.key==pygame.K_RIGHT):
                             crct_2=crct_2+1
                             t_2 = t_2 + t2_2
                             t1_2=t3_2=t4_2=0
                             #print("t2_2",t2_2)
                          keepplaying = False
                  if t2_2 != 0:
                    time.sleep(1.05-t2_2)
                  gameDisplay.fill(white)
                  pygame.display.update()
                  time.sleep(0.1)     
                  keepplaying = True

              elif s_1==2:
                  clock.tick(60)
                  gameDisplay.blit(image_2, (0,h/2))
                  pygame.display.update()
                  start_2_2= time.time()
                  while (time.time()-start_2_2)<1 and keepplaying:
                    clock.tick(60)
                    for event in pygame.event.get():
                      if event.type == pygame.QUIT:
                          pygame.quit() 
                          sys.exit()

                      if event.type==pygame.KEYDOWN:
                          end_2_2=time.time()
                          t2_2=end_2_2-start_2_2
                          m=event.key
                          #print ("Joystick '",joysticks[event.joy].get_name(),"' button",m,"down.")
                          if (event.key==pygame.K_LEFT):
                            crct_2=crct_2+1
                            t_2 = t_2 + t2_2
                            t1_2=t3_2=t4_2=0
                            #print("t2_2",t2_2)
                          keepplaying = False
                  if t2_2 != 0:
                    time.sleep(1.05-t2_2)
                  gameDisplay.fill(white)
                  pygame.display.update()
                  time.sleep(0.1)     
                  keepplaying = True
                           
       
              
      elif s == 3:
             clock.tick(60)
             if s_1==1:
               gameDisplay.blit(image_1,(w/2,h-100))
               pygame.display.update()
               start_3_2= time.time()
               while (time.time()-start_3_2)<1 and keepplaying:
                  clock.tick(60)
                  for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit() 
                        sys.exit()

                    if event.type==pygame.KEYDOWN:
                        end_3_2=time.time()
                        t3_2=end_3_2-start_3_2
                        m=event.key
                        #print ("Joystick '",joysticks[event.joy].get_name(),"' button",m,"down.")
                        if (event.key==pygame.K_DOWN):
                           crct_2=crct_2+1
                           t_2 = t_2 + t3_2
                           t2_2=t1_2=t4_2=0
                           #print("t3_2",t3_2)
                        keepplaying = False
               if t3_2 != 0:
                  time.sleep(1.05-t3_2)
               gameDisplay.fill(white)
               pygame.display.update()
               time.sleep(0.1)     
               keepplaying = True
               
             elif s_1==2:
               clock.tick(60)
               gameDisplay.blit(image_2,(w/2,h-100))
               pygame.display.update()
               start_3_2= time.time()
               while (time.time()-start_3_2)<1 and keepplaying:
                  clock.tick(60)
                  for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit() 
                        sys.exit()

                    if event.type==pygame.KEYDOWN:
                        end_3_2=time.time()
                        t3_2=end_3_2-start_3_2
                        m=event.key
                        #print ("Joystick '",joysticks[event.joy].get_name(),"' button",m,"down.")
                        if (event.key==pygame.K_LEFT):
                          crct_2=crct_2+1
                          t_2 = t_2 + t3_2
                          t2_2=t1_2=t4_2=0
                          #print("t3_2",t3_2)
                        keepplaying = False
               if t3_2 != 0:
                  time.sleep(1.05-t3_2)
               gameDisplay.fill(white)
               pygame.display.update()
               time.sleep(0.1)     
               keepplaying = True
         
              
      elif s == 4:
             clock.tick(60)
             if s_1==1:
                   gameDisplay.blit(image_1,(w-100,h/2))
                   pygame.display.update()
                   start_4_2= time.time()
                   while (time.time()-start_4_2)<1 and keepplaying:
                      clock.tick(60)
                      for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit() 
                            sys.exit()

                        if event.type==pygame.KEYDOWN:
                            end_4_2=time.time()
                            t4_2=end_4_2-start_4_2
                            m=event.key
                            #print ("Joystick '",joysticks[event.joy].get_name(),"' button",m,"down.")
                            if (event.key==pygame.K_RIGHT):
                               crct_2=crct_2+1
                               t_2 = t_2 + t4_2
                               t2_2=t3_2=t1_2=0
                               #print("t4_2",t4_2)
                            keepplaying = False
                        
                   if t4_2 != 0:
                      time.sleep(1.05-t4_2)
                   gameDisplay.fill(white)
                   pygame.display.update()
                   time.sleep(0.1)     
                   keepplaying = True
             elif s_1==2:
                 clock.tick(60)
                 gameDisplay.blit(image_2,(w-100,h/2)) 
                 pygame.display.update()
                 start_4_2= time.time()
                 while (time.time()-start_4_2)<1 and keepplaying:
                    clock.tick(60)
                    for event in pygame.event.get():
                      if event.type == pygame.QUIT:
                          pygame.quit() 
                          sys.exit()

                      if event.type==pygame.KEYDOWN:
                           end_4_2=time.time()
                           t4_2=end_4_2-start_4_2
                           m=event.key
                           #print ("Joystick '",joysticks[event.joy].get_name(),"' button",m,"down.")
                           if (event.key==pygame.K_LEFT):
                              crct_2=crct_2+1
                              t_2 = t_2 + t4_2
                              t2_2=t3_2=t1_2=0
                              #print("t4_2",t4_2)
                           keepplaying = False
                 if t4_2 != 0:
                    time.sleep(1.05-t4_2)
                 gameDisplay.fill(white)
                 pygame.display.update()
                 time.sleep(0.1)     
                 keepplaying = True
     
                     
  #print("time_2=",t_2)
  incorrect_2=10 - crct_2
  if crct_2==0:
    mean_time_2=0
  else:
    mean_time_2=t_2/crct_2
  gameDisplay.fill(white)
  screen_text = font.render(str(s1+str(crct_2)), True,blue)
  gameDisplay.blit(screen_text, [550,100])
  pygame.display.update()
  screen_text = font.render(str(s2+str(incorrect_2)), True,blue)
  gameDisplay.blit(screen_text, [550,200])
  pygame.display.update()
  screen_text = font.render(str(s3+str(mean_time_2)+s4), True,blue)
  gameDisplay.blit(screen_text, [550,300])
  pygame.display.update()

  print(f"Level 3: \n Correct: {crct_2} \n Incorrect: {incorrect_2} \n Mean Time: {mean_time_2} \n \n")

#           print(crct, crct_1, crct_2)
  
  tc=crct+crct_1+crct_2
  #print tc
  rt= 0.333* (mean_time+mean_time_1+mean_time_2)

  print(f"Overall Average Response Time: {rt}")
  #print rt


  # vrt_csv_fields = ['Mean Response Time', 'Level 1 MRT', 'Level 1 Correct Responses', 'Level 2 MRT', 'Level 2 Correct Responses', 'Level 3 MRT', 'Level 3 Correct Responses']
  # vrt_csv_filename = "vrt_results.csv" 
  # vrt_combined_results = [rt, mean_time, crct, mean_time_1, crct_1, mean_time_2, crct_2]

  # with open(vrt_csv_filename, "w", newline="") as file:
  
  vrt_combined_results = [rt, mean_time, crct, mean_time_1, crct_1, mean_time_2, crct_2]
  writer.writerow(vrt_combined_results)

  ############################################################################################################################
  ############################################################################################################################

  screen_text = font.render("The test is complete. Press Analyze for Results", True,black)
  gameDisplay.blit(screen_text, [80,400])
  pygame.display.update()
  time.sleep(1)
 # return oo
##    pygame.quit()
##    quit()


vrt_csv_fields = ['Mean Response Time', 'Level 1 MRT', 'Level 1 Correct Responses', 'Level 2 MRT', 'Level 2 Correct Responses', 'Level 3 MRT', 'Level 3 Correct Responses']
vrt_csv_filename = "vrt_results.csv" 


with open(vrt_csv_filename, "w", newline="") as file:
   writer = csv.writer(file)
   writer.writerow(vrt_csv_fields)

   for i in range (1, 11):
      gameLoop()


# gameLoop()
#print (oo)
#return oo
pygame.quit()

