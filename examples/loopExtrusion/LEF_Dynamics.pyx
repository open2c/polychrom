#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=True

import numpy as np

cimport numpy as np

import cython

cimport cython


cdef extern from "<stdlib.h>":
    double drand48()   

cdef cython.double randnum():
    return drand48()


cdef class LEFTranslocatorDirectional(object):
    cdef int N
    cdef int M
    cdef cython.double [:] emission
    cdef cython.double [:] stallLeft
    cdef cython.double [:] stallRight
    cdef cython.double [:] stallFalloff
    cdef cython.double [:] falloff
    cdef cython.double [:] pause
    cdef cython.double [:] cumEmission
    cdef cython.long [:] LEFs1
    cdef cython.long [:] LEFs2
    cdef cython.long [:] stalled1 
    cdef cython.long [:] stalled2
    cdef cython.long [:] occupied 
    
    cdef int maxss
    cdef int curss
    cdef cython.long [:] ssarray  
 
    
    def __init__(self, emissionProb, deathProb, stallProbLeft, stallProbRight, pauseProb, stallFalloffProb,  numLEF):
        emissionProb[0] = 0
        emissionProb[len(emissionProb)-1] = 0
        emissionProb[stallProbLeft > 0.9] = 0        
        emissionProb[stallProbRight > 0.9] = 0        
        
        self.N = len(emissionProb)
        self.M = numLEF
        self.emission = emissionProb
        self.stallLeft = stallProbLeft
        self.stallRight = stallProbRight
        self.falloff = deathProb
        self.pause = pauseProb
        cumem = np.cumsum(emissionProb)
        cumem = cumem / float(cumem[len(cumem)-1])
        self.cumEmission = np.array(cumem, np.double)
        self.LEFs1 = np.zeros((self.M), int)
        self.LEFs2 = np.zeros((self.M), int)
        self.stalled1 = np.zeros(self.M, int)
        self.stalled2 = np.zeros(self.M, int)
        self.occupied = np.zeros(self.N, int)
        self.stallFalloff = stallFalloffProb
        self.occupied[0] = 1
        self.occupied[self.N - 1] = 1
        self.maxss = 1000000
        self.curss = 99999999

        for ind in xrange(self.M):
            self.birth(ind)


    cdef birth(self, cython.int ind):
        cdef int pos,i 
  
        while True:
            pos = self.getss()
            if pos >= self.N - 1:
                print "bad value", pos, self.cumEmission[len(self.cumEmission)-1]
                continue 
            if pos <= 0:
                print "bad value", pos, self.cumEmission[0]
                continue 
 
            
            if self.occupied[pos] == 1:
                continue
            
            self.LEFs1[ind] = pos
            self.LEFs2[ind] = pos
            self.occupied[pos] = 1
            
            if (pos < (self.N - 3)) and (self.occupied[pos+1] == 0):
                if randnum() > 0.5:                  
                    self.LEFs2[ind] = pos + 1
                    self.occupied[pos+1] = 1
            
            return

    cdef death(self):
        cdef int i 
        cdef double falloff1, falloff2 
        cdef double falloff 
         
        for i in xrange(self.M):
            if self.stalled1[i] == 0:
                falloff1 = self.falloff[self.LEFs1[i]]
            else: 
                falloff1 = self.stallFalloff[self.LEFs1[i]]
            if self.stalled2[i] == 0:
                falloff2 = self.falloff[self.LEFs2[i]]
            else:
                falloff2 = self.stallFalloff[self.LEFs2[i]]              
            
            falloff = max(falloff1, falloff2)
            if randnum() < falloff:                 
                self.occupied[self.LEFs1[i]] = 0
                self.occupied[self.LEFs2[i]] = 0
                self.stalled1[i] = 0
                self.stalled2[i] = 0
                self.birth(i)
    
    cdef int getss(self):
    
        if self.curss >= self.maxss - 1:
            foundArray = np.array(np.searchsorted(self.cumEmission, np.random.random(self.maxss)), dtype = np.long)
            self.ssarray = foundArray
            #print np.array(self.ssarray).min(), np.array(self.ssarray).max()
            self.curss = -1
        
        self.curss += 1         
        return self.ssarray[self.curss]
        
        

    cdef step(self):
        cdef int i 
        cdef double pause
        cdef double stall1, stall2 
        cdef int cur1
        cdef int cur2 
        for i in range(self.M):            
            stall1 = self.stallLeft[self.LEFs1[i]]
            stall2 = self.stallRight[self.LEFs2[i]]
                                    
            if randnum() < stall1: 
                self.stalled1[i] = 1
            if randnum() < stall2: 
                self.stalled2[i] = 1

                         
            cur1 = self.LEFs1[i]
            cur2 = self.LEFs2[i]
            
            if self.stalled1[i] == 0: 
                if self.occupied[cur1-1] == 0:
                    pause1 = self.pause[self.LEFs1[i]]
                    if randnum() > pause1: 
                        self.occupied[cur1 - 1] = 1
                        self.occupied[cur1] = 0
                        self.LEFs1[i] = cur1 - 1
            if self.stalled2[i] == 0:                
                if self.occupied[cur2 + 1] == 0:                    
                    pause2 = self.pause[self.LEFs2[i]]
                    if randnum() > pause2: 
                        self.occupied[cur2 + 1] = 1
                        self.occupied[cur2] = 0
                        self.LEFs2[i] = cur2 + 1
        
    def steps(self,N):
        cdef int i 
        for i in xrange(N):
            self.death()
            self.step()
            
    def getOccupied(self):
        return np.array(self.occupied)
    
    def getLEFs(self):
        return np.array(self.LEFs1), np.array(self.LEFs2)
        
        
    def updateMap(self, cmap):
        cmap[self.LEFs1, self.LEFs2] += 1
        cmap[self.LEFs2, self.LEFs1] += 1

    def updatePos(self, pos, ind):
        pos[ind, self.LEFs1] = 1
        pos[ind, self.LEFs2] = 1



