# Maja Machine Learning Framework
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.


import time
from multiprocessing import Process, Queue
from threading import Thread

from mmlf.framework.observables import FloatStreamObservable
        
class ObservableProxy(object):
    
    def __init__(self, observable, manager):
        observable.addObserver(lambda *args: self.update(*args))
        
        self.title = observable.title
        
        self.observerID = manager.Value('i', -1)
        
    def setObserverQueue(self, observerQueue):
        self.observerQueue = observerQueue        
        
    def addObserver(self, observerID):
        self.observerID.value = observerID
            
    def update(self, *args):
        self.observerQueue.put((self.observerID.value, args))
        
class ObserverDispatcher(Thread):
    
    def __init__(self, queue):
        super(ObserverDispatcher, self).__init__()
        
        self.queue = queue
        
        self.id = 0
        
        self.observers = {}
    
    def stop(self):
        self.running = False
    
    def connectObserverToObservable(self, observable, observer):
        self.observers[self.id] = observer
        observable.addObserver(self.id)
        self.id += 1
    
    def run(self):
        self.running = True
        while self.running:
            try:
                id, arg =  self.queue.get(timeout=1.0)
            except: # To avoid blocking at the end of experiment 
                continue
            self.observers[id](*arg)
            

def remoteProcess(observableQueue, updateQueue):
    observable = FloatStreamObservable("Test", "Time", "Value")
    observableProxy = ObservableProxy(observable)
    observableQueue.put(observableProxy)
    
    time.sleep(1)
    observableProxy.setObserverQueue(updateQueue)
    
    for i in range(100):
        observable.addValue(i, i**2)
        time.sleep(1)


if __name__ == '__main__':    
    observableQueue = Queue()
    updateQueue = Queue()
    
    process = Process(target=remoteProcess, 
                      args=(observableQueue, updateQueue))
    process.start()
    
    observerDispatcher = ObserverDispatcher(updateQueue)
    observerDispatcher.start()
    
    observable = observableQueue.get()
    def observerFct(*value):
        print value
        
    observerDispatcher.connectObserverToObservable(observable, observerFct)
            
    process.join()

