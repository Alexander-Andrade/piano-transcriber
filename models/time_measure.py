import time


class TimeMeasure:
    
    def __init__(self):
        pass
    
    def start(msg = ""):
        self.start = time.time()
        if msg:
            print("starting {} {}".format(msg))
        
    def stop(msg = ""):
        end = time.time()
        elapsed = end - start
        return elapsed