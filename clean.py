import os

def func(cmd):
    print(cmd)
    os.system(cmd)

cmd = 'rm -r save_weights/*.model'
func(cmd)

cmd = 'rm -r save_weights/*.pickle'
func(cmd)

cmd = 'rm -r save_weights/*.png'
func(cmd)

cmd = 'rm -r imgs/*.png'
func(cmd)
