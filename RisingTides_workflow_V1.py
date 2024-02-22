"""
GNU General Public License, version 3 (GPLv3)

Copyright (C) 2023, Alec O'Dell and Maria C. Babiuc Hamilton  <babiuc@marshall.edu>
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
"""

"""
import all necessary libraries
"""

import os,sys
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
import scipy.constants as sc
import numpy as np
import math as mt
import cmath as cm
from scipy.optimize import *
from mpmath import *
from scipy.optimize import curve_fit

"""
geometric unit converter (G=c=1)
To convert geometric unit → SI, multiply by factor. 
To convert SI → geometric unit, divide by factor.
"""
#constants
Ms = 1.98840987e+30 # Solar mass
Ls = 3.828e+26 # Solar luminosity in W
c = sc.c      # Speed of light in vacuum
G = sc.G      # Gravitational constant
ϵ0 = sc.epsilon_0   # Vacuum electric permittivity

#Conversion Factors, mass, time, length and energy in units of mass
cM = Ms  #mass to solar mass factor (m)
cT = Ms*G*(c**(-3))  #mass to time factor (sec)
cL = Ms*G*(c**(-2))  #mass to length factor (m)
cE = Ms*c**2 #energy
cP = c**5/G # luminosity or Poynting Flux
cB = 1e+4*(c**3)/(G*Ms*((G*ϵ0)**(1/2))) # magnetic field in Gauss

def Run(m=2.8, q=1, rA=11.5, rB=11.5, R0=42, χf=0.69, χNS=2.1e-3,q1=0.7,q2=1.4187,q3=-0.499,f1=1.5251,f2=-1.1568,f3=0.1292,E0=0.048332,E2=0.000743,E4=0.000124,MD=0,MTPNf=3455):
    """initialize"""
    globals().update(locals())
    mA = m/(1+q)   #mass of star 1
    mB = m-mA #mass of star 2 (conservation)
    μ = ((mA*mB)/(mA+mB)) #reduced mass
    η = μ/m #symmetric mass ratio - unitless
    r0  = R0*cL/1000   #initial separation between the centers of mass in km
    globals().update(locals())
    """import computational module for inspiral"""
    #Import .py containing all modularized code
    import RisingTides_engine
    #import inspiral class 
    from RisingTides_engine import BBHinspiral as ins
    #initialize inspiral class with all global variables
    a=globals()
    ins.initialize(a)
    
    """compute x(t)=(v(t)/c)^2"""
    ins.XPN()
    globals().update(ins.retrievequantities())
    xPNarr=xPN.flatten()
    """Velociy as a Function of Time"""
    ins.vPN()
    globals().update(ins.retrievequantities())

    """Radial Separation as a function of time (in geometric units)"""
    ins.rPN()
    globals().update(ins.retrievequantities())

    """get time indexes"""
    ins.Times()
    globals().update(ins.retrievequantities())
    
    """PN angular frequency (from Keplers law in geometric units, x^3/2)"""
    ins.MΩPN()
    globals().update(ins.retrievequantities())

    """Orbital Phase"""
    ins.MΦPN()
    globals().update(ins.retrievequantities())

    """Building the PN Amplitude and Strain: attention, this is for BBH!"""
    ins.Waveform()
    globals().update(ins.retrievequantities())

    """Individual orbits, must use the orbital phase"""
    ins.Orbits()
    globals().update(ins.retrievequantities())

    """import and initalize merger class"""
    from RisingTides_engine import BBHmerger as mer
    a=globals()
    mer.initialize(a)
    globals().update(mer.retrievequantities())

    """Orbital Freq"""
    mer.MΩBoB()
    globals().update(mer.retrievequantities())

    """Phase"""
    mer.MΦBoB()
    globals().update(mer.retrievequantities())

    """Calculate the Strain"""
    mer.Waveform()
    globals().update(mer.retrievequantities())
    
    """import and initalize hybrid class"""
    from RisingTides_engine import BBHhybrid as hyb
    a=globals()
    hyb.initialize(a)
    globals().update(hyb.retrievequantities())

    """orbital phase"""
    hyb.Mϕhyb()
    globals().update(hyb.retrievequantities())

    """orbital freq"""
    hyb.Mωhyb()
    globals().update(hyb.retrievequantities())

    """hyb PN parameter"""
    hyb.Xhyb()
    globals().update(hyb.retrievequantities())

    """hybrid separation"""
    hyb.Rhyb()
    globals().update(hyb.retrievequantities())

    """Complete Orbits"""
    hyb.Orbits()
    globals().update(hyb.retrievequantities())

    """form the normalized hybrid amplitude"""
    hyb.Normalized_Amp()
    globals().update(hyb.retrievequantities())

    """shift the hybrid time array to set 0 at merger (the maximum amplitude)"""
    #Building hybrid time array 
    hyb.translated_time()
    globals().update(hyb.retrievequantities())

    """#Finally, form the strain of the gravitational waves"""
    hyb.Waveform()
    globals().update(hyb.retrievequantities())
    
    """import SXSBBH class for comparison with SXS data then initialize"""
    from RisingTides_engine import SXSBBH
    #upload the necessary data
    BBHAmp_raw = np.genfromtxt('AmpBBH0180_raw.dat', unpack=True)
    BBHϕ_raw = np.genfromtxt('ϕBBH0180_raw.dat', unpack=True)
    BBHh22_raw = np.genfromtxt('h22BBH0180_raw.dat', unpack=True)
    #reading the arrays separately
    BBHTime = BBHAmp_raw[0,:]
    BBHAmp = BBHAmp_raw[1,:]
    BBHϕ = BBHϕ_raw[1,:]
    BBHh22plus = BBHh22_raw[1,:]
    globals().update(locals())
    #initialize SXSBBH class
    a=globals()
    SXSBBH.initialize(a)
    globals().update(SXSBBH.retrievequantities())

    """align data"""
    SXSBBH.align()
    globals().update(SXSBBH.retrievequantities())

    """BBH freq and ang acccel"""
    SXSBBH.MωBBH()
    SXSBBH.MωdotBBH()
    globals().update(SXSBBH.retrievequantities())

    """import SXS strain"""
    BBHh22plus = BBHh22_raw[1,:]

    """location of matching frequencies BBH SXS and BBH hybrid"""
    iTrefω = np.abs(Mωhybarr - MωBBHarr[0]).argmin()
    ΔTref = MtarrhybM[iTrefω]-BBHTime[BBHi0]

    """import and initialize BBHvsBNS class"""
    from RisingTides_engine import BBHvsBNS
    #masses for SXS data 
    m1 =2.8
    m2 = 2.7
    #upload the SXS BNS0001 data, for G2 EOS
    BNS1Amp_raw = np.genfromtxt('AmpBNS0001_raw.dat', unpack=True)
    BNS1ϕ_raw = np.genfromtxt('ϕBNS0001_raw.dat', unpack=True)
    BNS1h22_raw = np.genfromtxt('h22BNS0001_raw.dat', unpack=True)
    #reading the arrays separately for BNS1
    BNS1Time = BNS1Amp_raw[0,:]
    BNS1Amp = BNS1Amp_raw[1,:]
    BNS1ϕ = BNS1ϕ_raw[1,:]
    BNS1h22plus = BNS1h22_raw[1,:]
    #upload the SXS BNS0002 data, for MP1b EOS
    BNS2Amp_raw = np.genfromtxt('AmpBNS0002_raw.dat', unpack=True)
    BNS2ϕ_raw = np.genfromtxt('ϕBNS0002_raw.dat', unpack=True)
    BNS2h22_raw = np.genfromtxt('h22BNS0002_raw.dat', unpack=True)
    #reading the arrays separately for BNS2
    BNS2Time = BNS2Amp_raw[0,:]
    BNS2Amp = BNS2Amp_raw[1,:]
    BNS2ϕ = BNS2ϕ_raw[1,:]
    BNS2h22plus = BNS2h22_raw[1,:]
    globals().update(locals())
    a=globals()
    BBHvsBNS.initialize(a)
    globals().update(BBHvsBNS.retrievequantities())

    """align BNS and BBH data"""
    BBHvsBNS.align()
    globals().update(BBHvsBNS.retrievequantities())

    """match phase and amp"""
    BBHvsBNS.matchphase()
    globals().update(BBHvsBNS.retrievequantities())
    BBHvsBNS.matchAmp()
    globals().update(BBHvsBNS.retrievequantities())

    """use to plot the Phase Difference BNS matched at start"""
    fact1 = BBH1Ampintarr[0]/BNS1Ampintarr[0]
    fact2 = BBH2Ampintarr[0]/BNS2Ampintarr[0]

    """keff for SXS BNS data"""
    keff1 = (3./16.)*(BNS1tΛ)
    keff2 = (3./16.)*(BNS2tΛ)

    """import and initialize tidalphase module"""
    from RisingTides_engine import TidalPhase as TP
    globals().update(locals())
    a=globals()
    TP.initialize(a)
    globals().update(TP.retrievequantities())

    """compute first fits"""
    TP.firstfits()
    globals().update(TP.retrievequantities())

    """curvefit coeff.'s of analytic correction to SXS data"""
    TP.curvefits()
    globals().update(TP.retrievequantities())

    """ratio used for: BNS1 coefficients fit to BNS 2 and BNS 2 coefficients fit to BNS1"""
    tΛratio = BNS2tΛ/BNS1tΛ

    """import and initalize tapering class"""
    from RisingTides_engine import Tapering as T
    globals().update(locals())
    a=globals()
    T.initialize(a)
    
    """taper"""
    T.taper()
    globals().update(T.retrievequantities())

    """import and initialize Tidal Amplitude class"""
    
    from RisingTides_engine import TidalAmp as TA
    globals().update(locals())
    a=globals()
    TA.initialize(a)
    globals().update(TA.retrievequantities())

    """compute amplitude correction and curvefit analytic correction to SXS data"""
    TA.AnaApprox()
    TA.curvefits()
    globals().update(TA.retrievequantities())

    #New Fit for Amplitude (fit to BNS 1)
    p=0
    Fit_AmpArr_BNS1=np.zeros(len(BNS1Tshift[0:Tcutoff1]))
    for i in AmpBNS1pm[0:mrgBNS1index]:
        Fit_AmpArr_BNS1[p]=i
    for i in AmpBNS1han:
        Fit_AmpArr_BNS1[p]=i
        
        #New Fit for Amplitude (fit to BNS 2)
    p=0
    Fit_AmpArr_BNS2=np.zeros(len(BNS2Tshift[0:Tcutoff2]))
    for i in AmpBNS2pm[0:mrgBNS2index]:
        Fit_AmpArr_BNS2[p]=i
    for i in AmpBNS2han:
        Fit_AmpArr_BNS2[p]=i

    """import and initialize ReconstructVars class"""
    from RisingTides_engine import ReconstructVars as RV
    globals().update(locals())
    a=globals()
    RV.initialize(a)
    globals().update(RV.retrievequantities())
                     
    """Calculate the individual orbits for BN1, BNS2 and hybrid BBH"""
    #Use the Heaviside corrected phase...
    #...for BNS 1/NR, star A
    xABNS1new = rABNS1new*np.cos(MϕBNS1new/(2))
    yABNS1new = rABNS1new*np.sin(MϕBNS1new/(2))
    xABNS1NRnew = rABNS1NRnew*np.cos(MϕBNS1NRnew/(2))
    yABNS1NRnew = rABNS1NRnew*np.sin(MϕBNS1NRnew/(2))
    #...for BNS 1/NR, star B
    xBBNS1new = rBBNS1new*np.cos(MϕBNS1new/(2))
    yBBNS1new = rBBNS1new*np.sin(MϕBNS1new/(2))
    xBBNS1NRnew = rBBNS1NRnew*np.cos(MϕBNS1NRnew/(2))
    yBBNS1NRnew = rBBNS1NRnew*np.sin(MϕBNS1NRnew/(2))
    #...for BNS 2/NR star A
    xABNS2new = rABNS2new*np.cos(MϕBNS2new/(2))
    yABNS2new = rABNS2new*np.sin(MϕBNS2new/(2))
    xABNS2NRnew = rABNS2NRnew*np.cos(MϕBNS2NRnew/(2))
    yABNS2NRnew = rABNS2NRnew*np.sin(MϕBNS2NRnew/(2))
    #...for BNS 2/NR star B
    xBBNS2new = rBBNS2new*np.cos(MϕBNS2new/(2))
    yBBNS2new = rBBNS2new*np.sin(MϕBNS2new/(2))
    xBBNS2NRnew = rBBNS2NRnew*np.cos(MϕBNS2NRnew/(2))
    yBBNS2NRnew = rBBNS2NRnew*np.sin(MϕBNS2NRnew/(2))
    #...for BBH to merger
    xABBH1hybarr = rABBH1hybarr*np.cos(MϕBBH1hybarr/(2))
    yABBH1hybarr = rABBH1hybarr*np.sin(MϕBBH1hybarr/(2))
    xBBBH1hybarr = rBBBH1hybarr*np.cos(MϕBBH1hybarr/(2))
    yBBBH1hybarr = rBBBH1hybarr*np.sin(MϕBBH1hybarr/(2))
    #merger circle
    rNS = RToM/2
    t = np.linspace(0, 2*np.pi, 100)
    xNS = rNS * np.cos(t)
    yNS = rNS * np.sin(t)

    """Calculate the strain for reconstructed BNS1, BNS2 and hybrid BBH"""
    # ...for BNS1
  
    hplusBNS1new = np.array(hplushyb(AmpBNS1new,MϕBNS1new)).flatten()
    hcrosBNS1new = np.array(hcroshyb(AmpBNS1new,MϕBNS1new)).flatten()

    hplusBNS1NRnew = np.array(hplushyb(AmpBNS1NRnew,MϕBNS1NRnew)).flatten()
    hcrosBNS1NRnew = np.array(hcroshyb(AmpBNS1NRnew,MϕBNS1NRnew)).flatten()

    # ...for BNS2
    hplusBNS2new = np.array(hplushyb(AmpBNS2new,MϕBNS2new)).flatten()
    hcrosBNS2new = np.array(hcroshyb(AmpBNS2new,MϕBNS2new)).flatten()

    hplusBNS2NRnew = np.array(hplushyb(AmpBNS2new,MϕBNS2new)).flatten()
    hcrosBNS2NRnew = np.array(hcroshyb(AmpBNS2new,MϕBNS2new)).flatten()

    # ...for hybrid BBH
    hplushybnew = np.array(hplushyb(AnaBBH1Aintarr,(AnaBBH1ϕintarr-AnaBBH1ϕintarr[0]))).flatten()
    hcroshybnew = np.array(hcroshyb(AnaBBH1Aintarr,(AnaBBH1ϕintarr-AnaBBH1ϕintarr[0]))).flatten()

    globals().update(locals())


    """Define Plotted Arrays as single variables"""

    #BNS (SXS) 1 phase array
    BNS1ϕArr=BNS1ϕintarr[:mrgBNS1index]-BNS1ϕintarr[0]
    
    #BNS (SXS) 2 phase 
    BNS2ϕArr=BNS2ϕintarr[:mrgBNS2index]-BNS2ϕintarr[0]
    
    #analytic pN tidal correction to phase aligned to BNS (SXS) 1 (up to merger)
    BNS1ϕTideCorPNArr=BNS1ϕTidePN[:mrgBNS1index]
    
    #analytic pN tidal correction to phase aligned to BNS (SXS) 2 (up to merger)
    BNS2ϕTideCorPNArr=BNS2ϕTidePN[:mrgBNS2index]
    
    #analytic NRTidal tidal correction to phase aligned to BNS (SXS) 1 (up to merger)
    BNS1ϕTideF1CorArr=BNS1ϕTideF1[:mrgBNS1index]
    
    #analytic NRTidal tidal correction to phase aligned to BNS (SXS) 2 (up to merger)
    BNS2ϕTideF1CorArr=BNS2ϕTideF1[:mrgBNS2index]
    
    #analytic baseline hybrid phase aligned to BNS (SXS) 1 (up to merger)
    AnaBBH1ϕArr=(AnaBBH1ϕintarr[:mrgBNS1index]-AnaBBH1ϕintarr[0])
    
    #analytic baseline hybrid phase aligned to BNS (SXS) 2 (up to merger)
    AnaBBH2ϕArr=(AnaBBH2ϕintarr[:mrgBNS2index]-AnaBBH2ϕintarr[0])
    
    #Numerical Relativity BBH (SXS) Amplitude
    BBH1AmpArr=BBH1Ampintarr[0:-2000]
    
    #Ratio of Rescaled BNS (SXS) 1 Amplitude to BBH (SXS) Amplitude
    BNS1_BBHratioArr=BNS1Ampintarr[0:-2000]*fact1/BBH1Ampintarr[0:-2000]
    
    #Ratio of Rescaled BNS (SXS) 2 Amplitude to BBH (SXS) Amplitude
    BNS2_BBHratioArr=BNS2Ampintarr*fact2/BBH2Ampintarr
    
    #Time array for Ratio of Rescaled BNS (SXS) 1 Amplitude to BBH (SXS) Amplitude
    Time_BNS1AmpRatio=BNS1Tshift[0:-2000]
    
    #Time Array for  Ratio of Rescaled BNS (SXS) 2 Amplitude to BBH (SXS) Amplitude
    Time_BNS2AmpRatio=BNS2Tshift+ΔTrefBNS12
    
    #BNS (SXS) 1 aligned plus polarization of strain
    BNS1hplus22=m1*BNS1h22plus[TrefBNS1:]
    
    #BNS (SXS) 2 aligned plus polarization of strain 
    BNS2hplus22=m2*BNS2h22plus[TrefBNS2:]
    
    #BBH (SXS) aligned plus polarization of strain
    BBHhplus22=BBHh22plus[BBHwi0:]
    
    #Time array (shifted SXS time arrays) for strain components of BNS (SXS) 1
    #Time_BNS1h=BNS2Time[TrefBNS2:] - BNS2Time[TrefBNS2]+ΔTrefBNS12
    
    #Time array (shifted SXS time arrays) for strain components of BNS (SXS) 2
    #Time_BNS2h=BBHTime[BBH1wi0:] - BBHTime[BBH1wiref]
    
    #Time array (shifted SXS time arrays) for strain components of BBH (SXS)
    #Time_BBHh=BNS1Time[TrefBNS1:-500] - BNS1Time[TrefBNS1]
    
    #Amplitude of the baseline BBH model  
    BBHbaselineAmparr=AnaBBH1Aintarr[:numT]
    
    #Amplitude of BNS (SXS) 1
    BNS1AmpArr=fact1*BNS1Ampintarr
    
    #Amplitude of BNS (SXS) 2
    BNS2AmpArr=fact2*BNS2Ampintarr
    #Time Array for  Amplitude of Baseline model
    Time_BBHAmp=BNS1Tshift[:numT]
    
    #Time Array for Amplitude of BNS (SXS) 1
    Time_BNS1Amp=BNS1Tshift
    
    #Time Array for Amplitude  of BNS (SXS) 2
    Time_BNS2Amp=BNS2Tshift+ΔTrefBNS12
    
    #Amplitudes for the pN approx until merger (BNS 1 comparison)
    PN_AmpArr_BNS1=AnaBBH1Aintarr[:mrgBNS1index]*(1.+BNS1PNATide[0:mrgBNS1index])
    
    #Amplitude for the NRTidal approx until merger (BNS1 comparison)
    NRT_AmpArr_BNS1=AnaBBH1Aintarr[:mrgBNS1index]*(1.+BNS1ATide[0:mrgBNS1index])
    
    #Amplitudes for the pN approx until merger (BNS 2 comparison)
    PN_AmpArr_BNS2=AnaBBH2Aintarr[:mrgBNS2index]*(1.+BNS2PNATide[0:mrgBNS2index])
    
    #Amplitude for the NRTidal approx until merger (BNS 2 comparison)
    NRT_AmpArr_BNS2=AnaBBH2Aintarr[:mrgBNS2index]*(1.+BNS2ATide[0:mrgBNS2index])
    
    #Time Array for Amplitude Arrays (BNS 1 comparison)
    Time_AmpComp_BNS1=BNS1Tshift[:mrgBNS1index]
    
    #Time Array for Amplitude Arrays (BNS 2 Comparison)
    Time_AmpComp_BNS2=BNS2Tshift[:mrgBNS2index]+ΔTrefBNS12
    
    #New Analytic Fit to Tidal Phase Correction for BNS (SXS) 1 
    BNS1ϕTideCor_Fit=BNS1ϕTidemrg[0:mrgBNS1index+4000]
    
    #New Analytic Fit to Tidal Phase Correction  for BNS (SXS) 2
    #BNS2ϕTideCor_Fit=BNS2ϕTidemrg_PN[0:mrgBNS2index+4000]
    BNS2ϕTideCor_Fit=BNS2ϕTidemrg[0:mrgBNS2index+4000]
    
    #True Tidal Phase Correction for BNS (SXS) 1
    BNS1ϕTrueTide=BNS1ϕTideNR[0:mrgBNS1index+4000]
    
    #True Tidal Phase Correction for BNS (SXS) 2
    BNS2ϕTrueTide=BNS2ϕTideNR[0:mrgBNS2index+4000]
    
    #Time Array for Tidal Phase Corrections for BNS (SXS) 1 (beyond merger)
    Time_BNS1ϕTide=BNS1Tshift[0:mrgBNS1index+4000]
    
    #Time Array for Tidal Phase Corrections for BNS (SXS) 2 (beyond merger)
    Time_BNS2ϕTide=BNS2Tshift[0:mrgBNS2index+4000]
    
    
    
    return globals().update(locals())

def retrieveresults():
    return globals()
