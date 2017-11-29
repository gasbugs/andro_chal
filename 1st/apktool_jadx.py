#python2.7
# command
# sudo python decompileApk.py /folder

import os
import sys
import subprocess
from multiprocessing import Pool

def decompile(walk_dir = "Challenge_andro_2nd_dataset", pool=16):
        command1 = []
        command2 = []
        command3 = []

    
        if walk_dir[-1] is not '/':
            walk_dir += '/'

        o_dir = walk_dir[:-1] + "_apktools/"
        os.system('mkdir '+ o_dir)
        #-------------------------------------decompile----------------------------
        print("start decompiling:", walk_dir)
        i = 0
        once = True
        for root, subdirs, files in os.walk(walk_dir):
            for file in files:
                if file.endswith(".apk") or file.endswith(".vir"):
                    i=i+1
                    #print(str(i)+"-"+file )
                    if file.endswith(".apk"):
                        o_file = root+file
                    else :
                        o_file = root+file.split('.')[0]+".apk"
                        os.system("mv {} {} ".format(root+file, o_file))

                    command1.append("apktool d {} -o {} -f > /dev/null".format(o_file, o_dir+file.split('.')[0]))
                    command2.append("unzip -n {} -d {} > /dev/null".format(o_file, o_dir+file.split('.')[0]))
                    command3.append('JAVA_OPTS="-Xmx16G" jadx -j 1 -d {} {}'.format(o_dir+file.split('.')[0]+'/out > /dev/null', o_dir+file.split('.')[0]+"/classes.dex"))
                    
        with Pool(pool) as p:
            i = 0
            len_command = len(command3)
            
            i = 0
            print('[*]start decompile apk to smali')
            
            for result in p.imap(os.system, command1):
                i += 1
                if i%10==0:
                    print('{}%           \r'.format(i/len_command))
            print('{}%           \r'.format(i/len_command))
               
            i = 0
            print('[*]start extract classes.dex from apk')
            
            for result in p.imap(os.system, command2):
                i += 1
                if i%10==0:
                    print('processing: {}%           \r'.format(i/len_command * 100))
            print('processing:{}%           \r'.format(i/len_command * 100))
             
            i = 0
            print('[*]start decompile dex to java')
            
            for result in p.imap(os.system, command3):
                i += 1
                if i%10==0:
                    print('processing:{}%           \r'.format(i/len_command * 100))
            print('processing:{}%           \r'.format(i/len_command * 100))
   
if __name__== "__main__":
    decompile(sys.argv[1], pool = 16)
    