"""
import all necessary libraries
"""
import os,sys
from scipy.integrate import odeint
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
import scipy.constants as sc
import numpy as np
import math as mt
import cmath as cm
from scipy.optimize import *
from scipy.optimize import curve_fit
from mpmath import *



"""global vars"""
BNS1tΛ, BNS2tΛ = 791, 1540

"""------------------------"""
"""------------------------"""

"""Function Bank:"""

#radial sep
def r_PN(x):
        rho0PN = 1.
        rho1PN = -1.+(1./3.)*η
        rho2PN = (1./36.)*η*(171.+4.*η)
        rho3PN = (-(24257./2520.)+(41./192.)*(np.pi)**2)*η \
            -(37./12.)*η**2+(2./81.)*η**3

        r_PN = ((rho0PN)*(np.power(x,-1))+(rho1PN)*((x)**(0)) \
                +(rho2PN)*((x)**(1))+(rho3PN)*((x)**(2)))
            
        return r_PN

#ϕTide function 
def AnaϕTide(x,n1,n1p5,n2,n2p5,n3,d1,d1p5,d2):
    kTeff = (3./32.)*(tΛ1+tΛ2) 
    
    num = (1. + n1*x + n1p5*x**(1.5) + n2*x**2 + n2p5*x**(2.5) + n3*x**3)
    den = (1. + d1*x + d1p5*x**(1.5) + d2*x**2)
    
    ϕTide = (13./(8.*η))*kTeff*(x**(2.5))*(num/den)
    
    return (ϕTide)
        
#ϕTide function for BNS1
def AnaϕTide1(x,n1,n1p5,n2,n2p5,n3,d1,d1p5,d2):
    tΛ1, tΛ2 = BNS1tΛ, BNS1tΛ
    kTeff = (3./32.)*(tΛ1+tΛ2) 
    
    num = (1. + n1*x + n1p5*x**(1.5) + n2*x**2 + n2p5*x**(2.5) + n3*x**3)
    den = (1. + d1*x + d1p5*x**(1.5) + d2*x**2)
    
    ϕTide = (13./(8.*η))*(kTeff)*(x**(2.5))*(num/den)

    return (ϕTide)

#ϕTide function for BNS2
def AnaϕTide2(x,n1,n1p5,n2,n2p5,n3,d1,d1p5,d2):
    tΛ1, tΛ2 = BNS2tΛ, BNS2tΛ
    kTeff = (3./32.)*(tΛ1+tΛ2) 
    
    num = (1. + n1*x + n1p5*x**(1.5) + n2*x**2 + n2p5*x**(2.5) + n3*x**3)
    den = (1. + d1*x + d1p5*x**(1.5) + d2*x**2)
    
    ϕTide = (13./(8.*η))*(kTeff)*(x**(2.5))*(num/den)

    return (ϕTide)


# determine the end frequency for tapering: eq. 11, https://arxiv.org/pdf/1804.02235.pdf
#merger frequency function 
def BNSωh(tΛ1,tΛ2):
    kTeff = (3./32.)*(tΛ1+tΛ2)
    
    ωh0 = 0.3586
    n1 = 3.354e-2
    n2 = 4.3153e-5
    d1 = 7.542e-2
    d2 = 2.236e-4
    
    num = (1 + n1*(kTeff) + n2*(kTeff)**2)
    den = (1 + d1*(kTeff) + d2*(kTeff)**2)
    
    ωh = ωh0*(num/den)
    
    return (ωh)

#Analytic tide amplitude function for BNS1
def AnaAmpTide1(x, d,p):
    tΛ1, tΛ2 = BNS1tΛ, BNS1tΛ
    kTeff = (3./32.)*(tΛ1+tΛ2)
    
    fact = (η/21.)*(np.sqrt((np.pi)/(5.)))*kTeff
    num = (x**6)*(672.-11.*x)
    den = (1.+(d*x**p))
    
    AmpTide = fact*num/den

    return (AmpTide)

#Analytic tide amplitude function for BNS2
def AnaAmpTide2(x, d,p):
    tΛ1, tΛ2 = BNS2tΛ, BNS2tΛ
    kTeff = (3./32.)*(tΛ1+tΛ2)
    
    fact = (η/21.)*(np.sqrt((np.pi)/(5.)))*kTeff
    num = (x**6)*(672.-11.*x)
    den = (1.+(d*x**p))
    
    AmpTide = fact*num/den

    return (AmpTide)

#zeroth order Luminosity 
#with factor of 1/2 to get rid of extra factor of 2 in ω_hyb
def L0hyb(v_array,ω_array):
    L0hyb=(4/15)*(mz**2)*((v_array)**2)*(((1/2)*ω_array)**4)
    return L0hyb

#Hybrid Luminosity
def LEMhyb(i):
    LEMhyb=1+((11/2)*(i**2))+(16*(i**4))+(35*(i**6))+(65*(i**8))  \
            +((217/2)*(i**10))+(168*(i**12))+(246*(i**14)) \
            +(345*(i**16))+((935/2)*(i**18))+((616)*(i**20))
    return LEMhyb


"""---------------------------------------------------"""
"""---------------------------------------------------"""
class BBHinspiral():
    
    def initialize(globalvars):
        globals().update(globalvars)
        # Schwarzschild radius
        rS = float((2.0*G*m*Ms/(c**2))/1000) 

        #initial and final radial separations in units of solar mass 
        Ri = 1000*(r0-(rA+rB))/cL #initial separation between the surfaces
        RT = 1000*(rA+rB)/cL       #radius where neutron stars touch

        """
        The light ring: this is an important quantity marking when the 
        PNN approximation breaks down and the matching to BoB happens
        It is given in terms of the Schwarzchild radius and the spin of the remnant
        Will take it's location around 2rS.
        """
        fLR = 2.20              # Light Ring factor: TUNE TO CHANGE THE MATCHING
        RLR = fLR*rS*1000/cL  # Light Ring radius

        """
        initial and final value of the frequencies and velocities from Kepler's Law 
        """
        Ω0_SI  = float((G*m*Ms/(r0*1000)**3)**(1/2))  #initial
        Ωt_SI  = float((G*m*Ms/((rA+rB)*1000)**3)**(1/2))  #when stars touch
        ΩLR_SI  = float((G*m*Ms/(2*rS*1000)**3)**(1/2))  #when stars touch
        # note that velocities do not depend on mass
        v0_SI = float((G*m*Ms/(r0*1000))**(1/2)) #initial
        vt_SI = float((G*m*Ms/((rA+rB)*1000))**(1/2)) ##when stars touch
        vLR_SI = float((G*m*Ms/(2*rS*1000))**(1/2)) ##at light ring

        #Notation for frequencies: 
        #capital letters for orbital quantities, lowercase for GW quantities
        Ω0  = cT*Ω0_SI #initial
        Ωt  = cT*Ωt_SI #when stars touch
        ΩLR = cT*ΩLR_SI #final/light ring

        """
        post Newtonian parameter/Integration Bounds (v/c)^2
        initial and at light ring
        """
        MΩ0  = m*Ω0 #float((R0)**(-3/2))   #initial
        MΩt  = m*Ωt #float((2*RNS)**(-3/2))  #when stars touch
        MΩLR = m*ΩLR #float((RLR)**(-3/2)) #final/light ring

        # rescaled radii
        R0oM = R0/m
        RioM = Ri/m
        RToM = RT/m
        RLRoM = RLR/m

        x0  = (v0_SI/c)**2   #initial
        xt  = (vt_SI/c)**2   #touch
        xLR = (vLR_SI/c)**2  #light ring

        TcSI = float((5/256)*((c**5)/(G**3)*(r0)**4/(η*((m*Ms)**3))))
        Tc = float((5/256)*((R0)**4/(η*(m**3))))
        Tt = float((5/256)*((RT)**4/(η*(m**3))))
        Tc_BBH = float((5/256)*((R0oM)**4/(η*(1**3))))
        
        dt= 0.1 #0.05 #time step
        t0=0 #initial time set at t=0

        #numberofelements=mt.ceil(MTPNf/(dt))  #number of time steps/elements in time array rounded up to nearest integer 
        #MtPNarr=np.linspace(t0,MTPNf,numberofelements) #time array created from 0 to Tbreak 
        MtPNarr=np.arange(t0,MTPNf+dt, dt, dtype=np.float64)
        
        return globals().update(locals())


    def XPN():

        def xiPN(x,MtPNarr):
            
            xi0PN = (64./5.)*η    
            
            xi0p5PN = 0.
            
            xi1PN = -(4./105.)*η*(743.+924.*η)
            
            xi1p5PN = 160.85*η
            
            xi2PN = (2.*η*(34103.+122949.*η+59472.*η**2))/2835.
            
            xi2p5PN = (-248.874-950.018*η)*η
            
            xi3PN = (1./4677750.)*η*(8951189331.+275.*(-56198689.+2045736.*(np.pi)**2)*η \
                     +36152325.*η**2-129475500.*η**3+976250880.*(np.log(x/x0)))

            
            xi3PNHT = η*(-22.2337-(109568./525.)*(np.euler_gamma) \
                       -(109568./525.)*(np.log(x**(3./2.)/x0)))
            
            xi3p5PN = (1./189.)*(np.pi)*η*(-2649.+143470.*η+147112.*η**2)
            
            xi4PN = -(388./1215.)*η*(-6191.63+29700.4*η-14837.5*η**2+(18929389./10864.)*η**3 \
                      +η**4+(np.euler_gamma)*(-1133.77+(369792./3395.)*η) \
                      +(-566.885+(20506176./3395.)*η)*(np.log(x)))
            
            xi4p5PN = (64./5.)*η*((205./6.)*(np.pi)**3*η+(np.pi)*((343801320119./745113600.) \
                       -(6848./105.)**(np.euler_gamma)-(51438847./48384.)*η+(42680611./145152.)*η**2 \
                       +(9731./1344.)*η**3-(13696./105.)*(np.log(2)))-(3424./105.)*(np.pi)*(np.log(x)))
            
            xi5PN = -(11567./4860.)*η*(-3862.58+10777.1*η-16459.*η**2+2765.87*η**3 \
                      -158.934*η**4+η**5+(np.euler_gamma)*(32.3594-628.209*η \
                      +13.3968*η**2)+(16.1797-5484.07*η-6163.97*η**2)*(np.log(x)))
            
            xi5p5PN = (98374./945.)*η*(433.679+(np.euler_gamma)*(20.4465-4.19889*η) \
                      -1411.45*η+1595.21*η**2-199.563*η**3+(np.pi)*η**4 \
                      +(10.2233-232.842*η)*(np.log(x)))
            
            xi6PN = -(155377./131220.)*η*(-140112.-1436.89*(np.euler_gamma)**2+511358.*η \
                      -361163.*η**2+82689.*η**3-7012.92*η**4+157.381*η**5 \
                      +η**6+(np.euler_gamma)*(26642.9-26708.6*η+720.722*η**2 \
                      +12.9198*η**3)+(13321.4-1436.89*(np.euler_gamma)-75688.9*η \
                      + 23085.7*η**2+26072.7*η**3)*(np.log(x))-359.221*(np.log(x))**2)
            
            xiPN = (((x)**5))*(xi0PN*((x)**0)+xi0p5PN*((x)**(1/2))+xi1PN*((x)**1) \
                             +xi1p5PN*((x)**(3/2))+xi2PN*((x)**2)+xi2p5PN*((x)**(5/2)) \
                             +(xi3PN+xi3PNHT)*((x)**3)+xi3p5PN*((x)**(7/2)) \
                             +xi4PN*((x)**4)+xi4p5PN*((x)**(9/2))+xi5PN*((x)**5) \
                             +xi5p5PN*((x)**(11/2))+xi6PN*((x)**6))    
            return (xiPN)
        

        xPN=odeint(xiPN,x0,MtPNarr,hmin=1e-32) #x(t) as a 1 column array
            
        xPNarr = np.array(xPN).flatten()
            
        return globals().update(locals())


    def vPN():
        vPNarr=xPNarr**(1/2)
        return globals().update(locals())

    def rPN():
        

        rPNarr = np.array(r_PN(xPN)).flatten()
        rdotarr=np.gradient(rPNarr,MtPNarr)

        return globals().update(locals())

    def Times():
                # find position where stars touch, and light ring
        """
        Change the domain if not found
        """
        t, tT, tLR = 0,0,0
        for t in range(len(MtPNarr)):
            if (0.999*(RT/m) < rPNarr[t] < 1.001*(RT/m)):
                tT = t
        #        print("stars touch index",(2*RNS/rPNarr[tT]), tT)

        for t in range(len(MtPNarr)):
            if (0.999*(RLR/m) < rPNarr[t] < 1.001*(RLR/m)):
                tLR = t
        #        print("light ring index",(RLR/rPNarr[tLR]), tLR)
        #the last value for the time found will be filled in           
        numLR = tLR
        numT = tT
        numL = len(MtPNarr)-1

        for t in range(len(MtPNarr)):
            if (0.999 < xPNarr[t]/xLR < 1.001):
                tX = t
        numX = tX
        
        return globals().update(locals())

        
    def MΩPN():
        MΩPN = (xPN)**(3/2)
        MΩPNarr = np.array(MΩPN).flatten()
        MΩdotPNarr = np.gradient(MΩPNarr,MtPNarr)
        return globals().update(locals())

    def MΦPN():
        # Create cubic interpolators for Ω(t) to be able to integrate it.
        #https://stackoverflow.com/questions/41860451/odeint-with-multiple-parameters-time-dependent
        MΩPNinterp = interp1d(MtPNarr, MΩPNarr, bounds_error=False, fill_value="extrapolate",kind = "cubic")

        # Create the function required by the ODE
        def MΦPNdot (y, t):
            MΦPNdot = MΩPNinterp(t)
            return MΦPNdot

        MΩ0 = (xPNarr[0])**(3/2)  # Initial Condition

        # Solving the ODE
        MΦPN = odeint(MΦPNdot, MΩ0, MtPNarr) #hmin=1e-32

        MΦPNarr = np.array(MΦPN).flatten()

        return globals().update(locals())
        
        
    def Waveform():
        """
        Building the PN Amplitude and Strain: attention, this is for BBH!
        """
        reAPNarr = (1./(rPNarr)-(rdotarr**2) + (rPNarr**2)*(MΩPNarr**2))
        imAPNarr = 2.*(rPNarr)*(rdotarr)*(MΩPNarr)

        hplusPNarr = -2*η*((reAPNarr)*np.cos(2*MΦPNarr) + (imAPNarr)*np.sin(2*MΦPNarr))
        hcrosPNarr = -2*η*((reAPNarr)*np.sin(2*MΦPNarr) - (imAPNarr)*np.cos(2*MΦPNarr))

        AmpPNarr = np.sqrt(((hplusPNarr)**2)+((hcrosPNarr)**2))
        
        return globals().update(locals())
        
        
    def Orbits():
        """
        Individual orbits, must use the orbital phase
        """
        rAPNarr = (mB/m)*rPNarr
        rBPNarr = -(mA/m)*rPNarr

        rAPNxarr = rAPNarr*np.cos(MΦPNarr)
        rAPNyarr = rAPNarr*np.sin(MΦPNarr)

        rBPNxarr = rBPNarr*np.cos(MΦPNarr)
        rBPNyarr = rBPNarr*np.sin(MΦPNarr)
        return globals().update(locals())

    
    def retrievequantities():
        b=globals()
        return b

"""-----------------------------------------------------"""

class BBHmerger():
    def initialize(globalvars):
        globals().update(globalvars)
        """system parameters"""
        Q22QNM = q1 + q2*(1 - χf)**(q3) #Quality Factor, Dominant 22 mode
        MfΩ22QNM = f1 +f2*(1 - χf)**(f3) #QNM Frequency Dominant 22 Mode 
        χeff = χNS*(1 - q)/(1 + q) 
        EGW = E0 + E2*χeff**(2) + E4*χeff**(4)
        Mf = (1 - EGW - MD)
        Ω22QNM = MfΩ22QNM/(Mf)
        τ = 2*(Q22QNM/Ω22QNM) #Damping Time
        ΩQNM = Ω22QNM/2 #final frequency of fundamental QNM
        """Set initial values as the final values of inspiral stage/PN expans. at light ring
        This can be tuned"""
        A0 = 1; t0 = 0
        MΩi = MΩPNarr[numLR] #assume light ring 4 for freq of PN
        MΩdoti = MΩdotPNarr[numLR] #assume light ring 4 for angular accel at light ring freq of PN
        """Initial value for the time of overlap of models """
        ti = t0-((τ/2)*np.log((((ΩQNM**4)-(MΩi**4))/(2*τ*(MΩi**3)*MΩdoti))-1))
        #we define the time array for the BoB
        tBoBmax = 250
        Nmin = int(ti/dt)
        tBoBmin = Nmin*dt
        Nmax = int((tBoBmax-tBoBmin)/dt)
        MtBoBarr = np.arange(tBoBmin, tBoBmax+dt, dt, dtype=np.float64)
        return globals().update(locals())
    
        """frequency of fundamental mode as funct of t"""
    def MΩBoB():
        kB = (((ΩQNM**4)-(MΩi**4))/(1-np.tanh((ti-t0)/τ)))
        def M_ΩBoB(t):
            M_ΩBoB = ((MΩi**4)+(kB*(np.tanh((t-t0)/τ)-np.tanh((ti-t0)/τ))))**(1/4)
            return M_ΩBoB
        MΩBoBarr = np.array(M_ΩBoB(MtBoBarr)).flatten()
        MΩdotBoBarr=np.gradient(MΩBoBarr,MtBoBarr)
        #find maximum for BoB angular acceleration
        maxMΩdotBoB = np.max(MΩdotBoBarr)
        # Get the indices of maximum element in numpy array
        tMΩdotmaxBoB = np.where(MΩdotBoBarr == maxMΩdotBoB) #location of max BoB amplitude
        return globals().update(locals()) 


    def MΦBoB(): 
        def M_ΦdotBoB (y, t):
            M_ΦdotBoB = ((MΩi**4)+(kB*(np.tanh((t-t0)/τ)-np.tanh((ti-t0)/τ))))**(1/4)
            return M_ΦdotBoB
        MΦBoBnum = odeint(M_ΦdotBoB, MΩi, MtBoBarr)
        MΦBoBnumarr = np.array(MΦBoBnum).flatten()
        return globals().update(locals())
    
    def Waveform():
        """Ang. Freq"""
        def M_ΩBoB(t):
            M_ΩBoB = ((MΩi**4)+(kB*(np.tanh((t-t0)/τ)-np.tanh((ti-t0)/τ))))**(1/4)
            return M_ΩBoB
        """The amplitude of the Weyl scalar Ψ4"""
        def Ψ4(t):
            Ψ4 = A0/np.cosh((t-t0)/τ)
            return Ψ4
        """The BoB amplitude for the dominant mode"""
        def AmpBoB(t):
            AmpBoB = ((Ψ4(t))/(M_ΩBoB(t))**2)
            return AmpBoB
        AmpBoBarr = np.array(AmpBoB(MtBoBarr)).flatten()
        hplusBoBarr = -AmpBoBarr*np.cos(2*MΦBoBnumarr)
        hcrosBoBarr = +AmpBoBarr*np.sin(2*MΦBoBnumarr)
        return globals().update(locals())
    

    def retrievequantities():
        b=globals()
        return b


"""-----------------------------------------------------"""

class BBHhybrid():
    def initialize(globalvars):
        globals().update(globalvars)
        
        """make hybrid time array"""
        #translate the time array from the PN approx/inspiral stage 
        MtarrshiftPN=np.zeros(numLR)
        p=0
        for e in MtPNarr[0:numLR]:
            MtarrshiftPN[p]=e - MtPNarr[numLR]
            p=p+1
        #translate the time array for freq. from the BoB approx/merger stage 
        MtarrshiftBoB=np.zeros(len(MtBoBarr))
        p=0
        for e in MtBoBarr:
            MtarrshiftBoB[p]= e - MtBoBarr[0]
            p=p+1
        #Building hybrid time array    
        Mtarrhyb=np.zeros(len(MtarrshiftPN)+len(MtarrshiftBoB))
        p=0
        for e in range(len(MtarrshiftPN)):
            Mtarrhyb[p]=MtarrshiftPN[e]
            p=p+1
        for e in range(0,len(MtarrshiftBoB)):
            Mtarrhyb[p]=MtarrshiftBoB[e]
            p=p+1
            
        return globals().update(locals())

    def Mϕhyb():
        # Fit the phase
        lastΦPN = MΦPNarr[numLR]
        firstΦNR = MΦBoBnumarr[0]
        ΔΦ = lastΦPN - firstΦNR
        #Building Hybrid Phase
        Mϕhybarr=np.zeros(len(Mtarrhyb))
        p=0
        for e in range(len(MtarrshiftPN)):
            Mϕhybarr[p]=2*MΦPNarr[e]
            p=p+1
        for e in range(0,len(MtarrshiftBoB)):
            Mϕhybarr[p]=2*(MΦBoBnumarr[e] + ΔΦ)
            p=p+1
        return globals().update(locals())

    def Mωhyb():
    # Fit the frequency
        lastΩPN = MΩPNarr[numLR]
        firstΩNR = MΩBoBarr[0]
        ΔΩ = lastΩPN - firstΩNR
    #Building Hybrid Angular Frequency 
        Mωhybarr=np.zeros(len(Mtarrhyb))
        p=0
        for e in range(len(MtarrshiftPN)):
            Mωhybarr[p]=2*MΩPNarr[e]
            p=p+1
        for e in range(len(MtarrshiftBoB)):
            Mωhybarr[p]=2*(MΩBoBarr[e]+ΔΩ)
            p=p+1
        return globals().update(locals())

    def Xhyb():
        #Calculate the hybrid PN parameter
        Xhybarr = (Mωhybarr/2.)**(2/3)
        return globals().update(locals())
        

    def Rhyb():
        def r_PN(x):
            rho0PN = 1.
            rho1PN = -1.+(1./3.)*η
            rho2PN = (1./36.)*η*(171.+4.*η)
            rho3PN = (-(24257./2520.)+(41./192.)*(np.pi)**2)*η \
                  -(37./12.)*η**2+(2./81.)*η**3

            r_PN = ((rho0PN)*(np.power(x,-1))+(rho1PN)*((x)**(0)) \
                    +(rho2PN)*((x)**(1))+(rho3PN)*((x)**(2)))
            
            return r_PN
        #Calculate the hybrid separation 
        Rhybarr = np.array(r_PN(Xhybarr)).flatten()
        return globals().update(locals())

    def Orbits():
        """
        Now will calculate the individual orbits for the whole evolution
        """
        rAhybarr = (mB/m)*Rhybarr
        rBhybarr = -(mA/m)*Rhybarr

        rAhybxarr = rAhybarr*np.cos(Mϕhybarr/(2))
        rAhybyarr = rAhybarr*np.sin(Mϕhybarr/(2))

        rBhybxarr = rBhybarr*np.cos(Mϕhybarr/(2))
        rBhybyarr = rBhybarr*np.sin(Mϕhybarr/(2))
        return globals().update(locals())

    def Normalized_Amp():

        #define the function for the amplitude
        def Amp(hplus,hcros):
            Amp=np.sqrt(((hplus)**2)+((hcros)**2))
            return Amp

        #find maximum for BoB amplitude
        maxAmpBoB = np.max(abs(AmpBoBarr))
        # Get the indices of maximum element in numpy array
        tAmaxBoB = np.where(AmpBoBarr == maxAmpBoB) #location of max BoB amplitude
        #form BoB norm amplitude 
        NormAmpBoBarr = AmpBoBarr/maxAmpBoB
        # find first BoB amplitude
        firstNormAmpBoB = NormAmpBoBarr[0]
        #find PN amplitude when stars touch
        lastAmpPN = AmpPNarr[numLR]
        #form PN norm amplitude
        NormAmpPNarr = AmpPNarr*firstNormAmpBoB/lastAmpPN 
        #form the normalized hybrid amplitude
        Amphybarr=np.zeros(len(Mtarrhyb))
        p=0
        for e in range(len(MtarrshiftPN)):
            Amphybarr[p]=NormAmpPNarr[e]
            p=p+1
        for e in range(0,len(MtarrshiftBoB)):
            Amphybarr[p]=NormAmpBoBarr[e]
            p=p+1
        #find maximum for BoB angular acceleration
        maxAhyb = np.max(Amphybarr)
        # Get the indices of maximum element in numpy array
        maxMtarrhyb = np.where(Amphybarr == maxAhyb) #location of max BoB amplitude
        numM = maxMtarrhyb[0][0]
        
        return globals().update(locals())

    def translated_time():
        MtarrhybM=np.zeros(len(Mtarrhyb))
        p=0
        for e in range(len(MtarrhybM)):
            MtarrhybM[p]=Mtarrhyb[e] - Mtarrhyb[numM]
            p=p+1
        numF = len(MtarrhybM)
        
        return globals().update(locals())
    
    def Waveform():
        #Finally, form the strain of the gravitational waves
        #define the functions
        def hplushyb(Amp,ϕ):
            hplushyb=Amp*np.cos(ϕ)
            return hplushyb

        def hcroshyb(Amp,ϕ):
            hcroshyb=-Amp*np.sin(ϕ)
            return hcroshyb
        #transform them into array to plot
        hplushybarr = np.array(hplushyb(Amphybarr,Mϕhybarr)).flatten()
        hcroshybarr = np.array(hcroshyb(Amphybarr,Mϕhybarr)).flatten()
        
        return globals().update(locals())
    

    def retrievequantities():
        b=globals()
        return b

"""--------------------------------------------------------"""

class SXSBBH():
    def initialize(globalvars):
        globals().update(globalvars)

        #location of max SXS BBH amplitude
        BBHAmax = np.max(BBHAmp)
        i = 0
        for i in range(len(BBHAmp)):
            if (BBHAmp[i] == BBHAmax):
                BBHiM = i
                break
            
        #location of first SXS point, tuned to the Analytic BBH start time location        
        for i in range(len(BBHTime)):
            if (0.9998 < BBHTime[i]/MtarrhybM[0] < 1.0001):
                BBHti0 = i
        


        #location of first SXS point, tuned to the Analytic BBH start frequency location        
        #calculating the BBH frequency
        BBHω=np.gradient(BBHϕ,BBHTime)
        for i in range(len(BBHTime)):
            if (0.9999 < BBHω[i]/Mωhybarr[0] < 1.0001):
                BBHωi0 = i
                
        return globals().update(locals())
    def MωBBH():
        MωBBHarr=np.gradient(BBHϕ[BBHi0:BBHiF],BBHTime[BBHi0:BBHiF])
        return globals().update(locals())
    
    def MωdotBBH():
        MωdotBBHarr=np.gradient(MωBBHarr,BBHTime[BBHi0:BBHiF])
        return globals().update(locals())

    def align():
        #location of last SXS point   
        BBHtF = BBHTime[-1]
        BBHiF = len(BBHTime)-1
        
        # pick the initial point for the SXS BBH array
        BBHi0 =   BBHti0 #BBHωi0 
        Δt0 = (BBHTime[BBHi0]- MtarrhybM[0])
        
        # difference in amplitude at the initial time
        BBHΔAmp = Amphybarr[0] - BBHAmp[BBHi0]
        return globals().update(locals())
        
    def retrievequantities():
        b=globals()
        return b

"""--------------------------------------------------------"""
class BBHvsBNS():
    #funct to find frequency at reference phase
    def dfunO2(fip1, fim1, h):
        return (fip1 - fim1)/(2*h)
    
    #funct to find frequency at reference phase
    def dfunO4(fip2, fip1, fim1, fim2, h):    
        return (-fip2 + 8*fip1 - 8*fim1 + fim2)/(12*h)
    
    def initialize(globalvars):
        globals().update(globalvars)
        
        #correct the time arrays with reference time - where initial data error become insignificant
        TrefBBH = 282
        TrefBNS1 = 1195 #1278
        TrefBNS2 = 985 # 959

        #must find frequency at reference phase
        #BNS1
        h1 = (BNS1Time[TrefBNS1+1]-BNS1Time[TrefBNS1-1])/2
        BNS1ωref_O2 = BBHvsBNS.dfunO2(BNS1ϕ[TrefBNS1+1], BNS1ϕ[TrefBNS1-1], h1)
        BNS1ωref_O4 = BBHvsBNS.dfunO4(BNS1ϕ[TrefBNS1+2], BNS1ϕ[TrefBNS1+1], BNS1ϕ[TrefBNS1-1], BNS1ϕ[TrefBNS1-2],h1)
        #BNS2
        h2 = (BNS2Time[TrefBNS2+1]-BNS2Time[TrefBNS2-1])/2
        BNS2ωref_O2 = BBHvsBNS.dfunO2(BNS2ϕ[TrefBNS2+1], BNS2ϕ[TrefBNS2-1], h2)
        BNS2ωref_O4 = BBHvsBNS.dfunO4(BNS2ϕ[TrefBNS2+2], BNS2ϕ[TrefBNS2+1], BNS2ϕ[TrefBNS2-1], BNS2ϕ[TrefBNS2-2],h2)
        
        #location of matching frequencies  BBH SXS point, tuned to the BNS location        
        for i in range(len(BBHω)):
            if (0.9999 < BBHω[i]/BNS1ωref_O4 < 1.001):
                BBH1ωiref = i

        for i in range(len(BBHω)):
            if (0.9999 < BBHω[i]/BNS2ωref_O4 < 1.001):
                BBH2ωiref = i

        BBH1Δϕref = BBHϕ[BBH1ωiref] - BNS1ϕ[TrefBNS1]
        BBH2Δϕref = BBHϕ[BBH2ωiref] - BNS2ϕ[TrefBNS2]
        BNS1Ampref = BNS1Amp/BNS1Amp[TrefBNS1]
        BBH1Ampωref = BBHAmp/BBHAmp[BBH1ωiref]
        BNS2Ampref = BNS2Amp/BNS2Amp[TrefBNS2]
        BBH2Ampωref = BBHAmp/BBHAmp[BBH2ωiref]
        return globals().update(locals())
    
    def align():
        #location of matching frequencies analytic BBH (point particle) point, tuned to the BNS location  

        for i in range(len(Mϕhybarr)):
            if (0.9999 < Mωhybarr[i]/BNS1ωref_O4 < 1.0001):
                Mωhybiref_BBH1 = i

        for i in range(len(Mϕhybarr)):
            if (0.9999 < Mωhybarr[i]/BNS2ωref_O4 < 1.0001):
                Mωhybiref_BBH2 = i
                
        #Making arrays out of the releveant subsections of the BNS NR arrays for convienence
        #will still need to "cut off" the ends of the arrays according to when our simulation ends
        #phase array to match to BNS 1
        MϕhybarrBNS1=Mϕhybarr[Mωhybiref_BBH1:]-Mϕhybarr[Mωhybiref_BBH1]
        
        #Corresponding time array
        MtarrhybBNS1=Mtarrhyb[Mωhybiref_BBH1:]-Mtarrhyb[Mωhybiref_BBH1]
        NR1Time=BNS1Time[TrefBNS1:-1]-BNS1Time[TrefBNS1]
        ϕNR1=BNS1ϕ[TrefBNS1:-1]-BNS1ϕ[TrefBNS1]
        
        #phase array to match to BNS 2
        MϕhybarrBNS2=Mϕhybarr[Mωhybiref_BBH2:]-Mϕhybarr[Mωhybiref_BBH2]
        
        #Corresponding time array
        MtarrhybBNS2=Mtarrhyb[Mωhybiref_BBH2:]-Mtarrhyb[Mωhybiref_BBH2]
        NR2Time=BNS2Time[TrefBNS2:-1]-BNS2Time[TrefBNS2]
        ϕNR2=BNS2ϕ[TrefBNS2:-1]-BNS2ϕ[TrefBNS2]

        
        #We must form the time arrays to match with the analytical BBH....
        
        #find the time array in the Analytic BBH corresponding to the BNS frequency, two ways:
        #location of first Analytic point, tuned to the BNS1 ref location        
        for i in range(len(MtarrhybM)):
            if (0.9999 < MtarrhybM[i]/BBHTime[BBH1ωiref] < 1.0001):
                numtBNS1ref = i
                
        #location of first Analytic point, tuned to the BNS2 ref location        
        for i in range(len(MtarrhybM)):
            if (0.9999 < MtarrhybM[i]/BBHTime[BBH2ωiref] < 1.0001):
                numtBNS2ref = i
        for i in range(len(Mωhybarr)):
            if (0.9999 < Mωhybarr[i]/BNS2ωref_O4 < 1.0001):
                numωBNS2ref = i
        for i in range(len(Mωhybarr)):
            if (0.9999 < Mωhybarr[i]/BNS1ωref_O4 < 1.0001):
                numωBNS1ref = i


        # the time match is better, there is a time mismatch between analytic and NR time arrays
        numBNS1ref = numtBNS1ref
        numBNS2ref = numtBNS2ref

        #scale the phase with the reference phase amplitude of BBH
        Ana1Δϕref = Mϕhybarr[numBNS1ref] - BNS1ϕ[TrefBNS1]
        Ana2Δϕref = Mϕhybarr[numBNS2ref] - BNS2ϕ[TrefBNS2]

        #scale the amplitudes with the reference phase amplitude of BBH
        Amphyb1ref = Amphybarr/Amphybarr[numBNS1ref]
        Amphyb2ref = Amphybarr/Amphybarr[numBNS2ref]        
        return globals().update(locals())

    def matchphase():
        #it's time to find the last point of the BNS time array
        # note that the final BBH time is smaller than the final BNS1 time. Should stop there!
        for i in range(len(BNS1Time)):    
            if (0.9999 < (BNS1Time[i]-BNS1Time[TrefBNS1])/(BBHTime[-1]-BBHTime[BBH1ωiref]) < 1.0001):
                TfinBNS1 = i

        #equally spaced time array for BNS1 interpolation
        # BNS1Time[TrefBNS1:TfinBNS1]; MtarrhybM[numωBNS1ref:], BBHTime[BBH1ωiref:]
        BNS1T0 = BNS1Time[TrefBNS1]
        BNS1Tf = BNS1Time[TfinBNS1] 
        BNS1Tarr = np.arange(BNS1T0,BNS1Tf+dt, dt, dtype=np.float64)
        

        #equally spaced time array for BNS1 interpolation
        # BNS2Time[TrefBNS2:-1]; MtarrhybM[numωBNS2ref:], BBHTime[BBH2ωiref:]
        BNS2T0 = BNS2Time[TrefBNS2]
        BNS2Tf = BNS2Time[-1] 
        BNS2Tarr = np.arange(BNS2T0,BNS2Tf, dt, dtype=np.float64)
       

        # note that the final BNS1 time is smaller than the final BBH time. Should stop there!
        for i in range(len(BBHTime)):    
            if (0.9999 < (BBHTime[i]-BBHTime[BBH2ωiref])/(BNS2Time[-1]-BNS2Time[TrefBNS2]) < 1.0001):
                TfinBBH2 = i

        #fitted arrays
        #time arrays for BNS1 interpolation
        BBH1Tarr = np.zeros(len(BNS1Tarr))
        p=0
        for e in range(len(BNS1Tarr)):
            BBH1Tarr[p]=BBHTime[BBH1ωiref]+p*dt
            p=p+1
             
        AnaBBH1Tarr = np.zeros(len(BNS1Tarr))
        p=0
        for e in range(len(BNS1Tarr)):
            AnaBBH1Tarr[p] = MtarrhybM[numBNS1ref]+p*dt
            p=p+1
        

        #time arrays for BNS2 interpolation
        BBH2Tarr = np.zeros(len(BNS2Tarr))
        p=0
        for e in range(len(BNS2Tarr)):
            BBH2Tarr[p]=BBHTime[BBH2ωiref]+p*dt
            p=p+1
        
            
        AnaBBH2Tarr = np.zeros(len(BNS2Tarr))
        p=0
        for e in range(len(BNS2Tarr)):
            AnaBBH2Tarr[p] = MtarrhybM[numBNS2ref]+p*dt
            p=p+1
        
        ##create a shifted time array just for convenience in plotting: 
        BNS1Tshift = np.zeros(len(BNS1Tarr))
        p=0
        for e in range(len(BNS1Tarr)):
            BNS1Tshift[p]=BNS1Tarr[e] - BNS1Tarr[0]
            p=p+1
            
        BNS2Tshift = np.zeros(len(BNS2Tarr))
        p=0
        for e in range(len(BNS2Tarr)):
            BNS2Tshift[p]=BNS2Tarr[e] - BNS2Tarr[0]
            p=p+1

        #now that we painstakely built the arrays, let's interpolate
        #time arrays for BNS1 interpolation
        BNS1ϕint = interp1d(BNS1Time[TrefBNS1:TfinBNS1],BNS1ϕ[TrefBNS1:TfinBNS1], bounds_error=False, fill_value="extrapolate",kind = "cubic")
        BNS1ϕintarr = np.array(BNS1ϕint(BNS1Tarr)).flatten()
        

        BBH1ϕint = interp1d(BBHTime[BBH1ωiref:-1],BBHϕ[BBH1ωiref:-1], bounds_error=False, fill_value="extrapolate",kind = "cubic")
        BBH1ϕintarr = np.array(BBH1ϕint(BBH1Tarr)).flatten()
        
        numBNS1fin = len(AnaBBH1Tarr)+numBNS1ref
        
        AnaBBH1ϕint = interp1d(MtarrhybM[numBNS1ref:numBNS1fin], Mϕhybarr[numBNS1ref:numBNS1fin], bounds_error=False, fill_value="extrapolate",kind = "cubic")
        AnaBBH1ϕintarr = np.array(AnaBBH1ϕint(AnaBBH1Tarr)).flatten()


        #now that we painstakely built the arrays, let's interpolate
        #time arrays for BNS2 interpolation
        BNS2ϕint = interp1d(BNS2Time[TrefBNS2:-1],BNS2ϕ[TrefBNS2:-1], bounds_error=False, fill_value="extrapolate",kind = "cubic")
        BNS2ϕintarr = np.array(BNS2ϕint(BNS2Tarr)).flatten()
        

        BBH2ϕint = interp1d(BBHTime[BBH2ωiref:TfinBBH2],BBHϕ[BBH2ωiref:TfinBBH2], bounds_error=False, fill_value="extrapolate",kind = "cubic")
        BBH2ϕintarr = np.array(BBH2ϕint(BBH2Tarr)).flatten()
        

        numBNS2fin = len(AnaBBH2Tarr)+numBNS2ref
        
        AnaBBH2ϕint = interp1d(MtarrhybM[numBNS2ref:numBNS2fin], Mϕhybarr[numBNS2ref:numBNS2fin], bounds_error=False, fill_value="extrapolate",kind = "cubic")
        AnaBBH2ϕintarr = np.array(AnaBBH2ϕint(AnaBBH2Tarr)).flatten()
        ΔTrefBNS12 = (BNS2Time[TrefBNS2]-BNS1Time[TrefBNS1])
        ΔϕrefBNS12 = BBHϕ[BBH2ωiref]-BBHϕ[BBH1ωiref]
        return globals().update(locals())

    def matchAmp():
        #repeat the procedure for amplitudes
        #time arrays for BNS1 interpolation
        BBH1Ampint = interp1d(BBHTime[BBH1ωiref:-1],BBHAmp[BBH1ωiref:-1], bounds_error=False, fill_value="extrapolate",kind = "cubic")
        BBH1Ampintarr = np.array(BBH1Ampint(BBH1Tarr)).flatten()
        

        BNS1Ampint = interp1d(BNS1Time[TrefBNS1:TfinBNS1],BNS1Amp[TrefBNS1:TfinBNS1], bounds_error=False, fill_value="extrapolate",kind = "cubic")
        BNS1Ampintarr = np.array(BNS1Ampint(BNS1Tarr)).flatten()
        

        AnaBBH1Aint = interp1d(MtarrhybM[numBNS1ref:numBNS1fin], Amphybarr[numBNS1ref:numBNS1fin], bounds_error=False, fill_value="extrapolate",kind = "cubic")
        AnaBBH1Aintarr = np.array(AnaBBH1Aint(AnaBBH1Tarr)).flatten()
        

        #time arrays for BNS2 interpolation
        BBH2Ampint = interp1d(BBHTime[BBH2ωiref:TfinBBH2],BBHAmp[BBH2ωiref:TfinBBH2], bounds_error=False, fill_value="extrapolate",kind = "cubic")
        BBH2Ampintarr = np.array(BBH2Ampint(BBH2Tarr)).flatten()
        

        BNS2Ampint = interp1d(BNS2Time[TrefBNS2:-1],BNS2Amp[TrefBNS2:-1], bounds_error=False, fill_value="extrapolate",kind = "cubic")
        BNS2Ampintarr = np.array(BNS2Ampint(BNS2Tarr)).flatten()
        

        AnaBBH2Aint = interp1d(MtarrhybM[numBNS2ref:numBNS2fin], Amphybarr[numBNS2ref:numBNS2fin], bounds_error=False, fill_value="extrapolate",kind = "cubic")
        AnaBBH2Aintarr = np.array(AnaBBH2Aint(AnaBBH2Tarr)).flatten()

        ΔArefBNS12 = (BBHAmp[BBH2ωiref]-BBHAmp[BBH1ωiref])/BBH1Ampintarr[0]
        return globals().update(locals())
        
        
    def retrievequantities():
        b=globals()
        return b


"""--------------------------------------------------------"""
class TidalPhase():
    
    def initialize(globalvars):
        
        globals().update(globalvars)
        
        # now we can take the differences between the Amplitudes to find the tidal correction
        
        ΔBNS1AmpTideNR = ((BNS1Ampintarr/BNS1Ampintarr[0])-(BBH1Ampintarr/BBH1Ampintarr[0]))/BBH1Ampintarr
        ΔBNS2AmpTideNR = ((BNS2Ampintarr/BNS2Ampintarr[0])-(BBH2Ampintarr/BBH2Ampintarr[0]))/BBH2Ampintarr
        
        #find the merger location as the position where the change in amplitude is maximum
        mrgBBH1index=np.where(BBH1Ampintarr==np.max(BBH1Ampintarr))[0][0]
        mrgBNS1index=np.where(BNS1Ampintarr==np.max(BNS1Ampintarr))[0][0]
        mrgBNS2index=np.where(ΔBNS2AmpTideNR==np.max(ΔBNS2AmpTideNR))[0][0]
        """***compare the above line of code to the one above it***"""
        
        # now we can take the differences between the phases to find the tidal correction
        BNS1ϕTideNR = (BNS1ϕintarr-BNS1ϕintarr[0])-(BBH1ϕintarr-BBH1ϕintarr[0])
        BNS1ϕTideAna = (BNS1ϕintarr-BNS1ϕintarr[0])-(AnaBBH1ϕintarr-AnaBBH1ϕintarr[0])

        BNS2ϕTideNR = (BNS2ϕintarr-BNS2ϕintarr[0])-(BBH2ϕintarr-BBH2ϕintarr[0])
        BNS2ϕTideAna = (BNS2ϕintarr-BNS2ϕintarr[0])-(AnaBBH2ϕintarr-AnaBBH2ϕintarr[0])

        BNS1ωh = BNSωh(BNS1tΛ, BNS1tΛ)
        BNS2ωh = BNSωh(BNS2tΛ, BNS2tΛ)

        # find the analytical location of the merger AnaTarray,BNS2ϕinterparr
        BNS1ωarr=np.gradient(BNS1ϕintarr, BNS1Tarr)

        BNS2ωarr=np.gradient(BNS2ϕintarr, BNS2Tarr)

        # find the merger frequency, pn param and velocity
        #Calculate the merger frequency and PN parameter
        X1mrg = (BNS1ωh/2.)**(2/3)
        X2mrg = (BNS2ωh/2.)**(2/3)
        #Calculate the merger velocity
        V1mrg = (X1mrg)**(1/2)
        V2mrg = (X2mrg)**(1/2)

        #location of estimated merger in the Xhybarr array    
        for i in range(len(Xhybarr)):
            if (0.9999 < Xhybarr[i]/X1mrg < 1.001):
                numX1mrg = i
        
        for i in range(len(Xhybarr)):
            if (0.9999 < Xhybarr[i]/X2mrg < 1.001):
                numX2mrg = i

        #the PN 
        n1_PN = 3115/624
        n1p5_PN = 0 #-5*(np.pi)/2
        n2_PN = 0 #28024205/1100736
        n2p5_PN = 0 #-4283*(np.pi)/312
        n3_PN = 0
        d1_PN = 0
        d1p5_PN = 0
        d2_PN = 0
        coeffs_PN=np.array([n1_PN,n1p5_PN,n2_PN,n2p5_PN,n3_PN,d1_PN,d1p5_PN,d2_PN])

        #first fit, from 1804.02235.pdf
        c1_f1 = 1817/364
        n1_f1 = -17.941
        n1p5_f1 = 57.983 
        n2_f1 = -298.876 
        n2p5_f1 = 964.192
        n3_f1 = -936.844
        d1p5_f1 = 43.446
        d1_f1 = n1_f1 - c1_f1
        d2_f1 = 0.0
        coeffs_F1=np.array([n1_f1, n1p5_f1, n2_f1, n2p5_f1, n3_f1, d1_f1, d1p5_f1, d2_f1])

        #second first fit, from 1905.06011.pdf
        c1_f2 = 3115/624
        c1p5_f2 = -5*(np.pi)/2
        c2_f2 = 28024205/1100736
        c2p5_f2 = -4283*(np.pi)/312
        n2p5_f2 = 312.48173 
        n3_f2 = -342.15498 
        d1_f2 = -20.237200 
        d2_f2 = -5.361630
        n1_f2   = c1_f2 + d1_f2
        n1p5_f2 = (c1_f2*c1p5_f2 - c2p5_f2 - c1p5_f2*d1_f2 + n2p5_f2)/c1_f2
        n2_f2   = c2_f2 + c1_f2*d1_f2 + d2_f2
        d1p5_f2 = - (c2p5_f2 + c1p5_f2*d1_f2 - n2p5_f2)/c1_f2
        coeffs_F2=np.array([n1_f2, n1p5_f2, n2_f2, n2p5_f2, n3_f2, d1_f2, d1p5_f2, d2_f2])

        numBNS1off = len(BNS1Tshift)-len(Xhybarr[numBNS1ref:numX1mrg])
        numBNS2off =  len(BNS2Tshift)-len(Xhybarr[numBNS2ref:numX2mrg])
        return globals().update(locals())

    def firstfits():
        
        # PN Tide, into the postmerger, BNS 1 
        BNS1ϕTidePN = AnaϕTide1(Xhybarr[numBNS1ref:numBNS1fin],*coeffs_PN) - AnaϕTide1(Xhybarr[numBNS1ref],*coeffs_PN)
        BNS1ϕTidePNarr = np.array(BNS1ϕTidePN).flatten()
        
        
        # Fit 1 Tide, into the postmerger BNS 1
        BNS1ϕTideF1 = AnaϕTide1(Xhybarr[numBNS1ref:numBNS1fin],*coeffs_F1) - AnaϕTide1(Xhybarr[numBNS1ref],*coeffs_F1)
        BNS1ϕTideF1arr = np.array(BNS1ϕTideF1).flatten()

        # Fit 2 Tide, into the postmerger BNS 1
        BNS1ϕTideF2 = AnaϕTide1(Xhybarr[numBNS1ref:numBNS1fin],*coeffs_F2) - AnaϕTide1(Xhybarr[numBNS1ref],*coeffs_F2)
        BNS1ϕTideF2arr = np.array(BNS1ϕTideF2).flatten()

        # PN Tide, into the postmerger, BNS 2 
        BNS2ϕTidePN = AnaϕTide2(Xhybarr[numBNS2ref:numBNS2fin],*coeffs_PN) - AnaϕTide2(Xhybarr[numBNS2ref],*coeffs_PN)
        BNS2ϕTidePNarr = np.array(BNS2ϕTidePN).flatten()


        # Fit 1 Tide, into the postmerger BNS 2
        BNS2ϕTideF1 = AnaϕTide2(Xhybarr[numBNS2ref:numBNS2fin],*coeffs_F1) - AnaϕTide2(Xhybarr[numBNS2ref],*coeffs_F1)
        BNS2ϕTideF1arr = np.array(BNS2ϕTideF1).flatten()

        # Fit 2 Tide, up to merger BNS 2
        BNS2ϕTideF2mrg = AnaϕTide2(Xhybarr[numBNS2ref:numX2mrg],*coeffs_F2) - AnaϕTide2(Xhybarr[numBNS2ref],*coeffs_F2)
        BNS2ϕTideF2mrgarr = np.array(BNS2ϕTideF2mrg).flatten()

        # Fit 2 Tide, into the postmerger BNS 2
        BNS2ϕTideF2 = AnaϕTide2(Xhybarr[numBNS2ref:numBNS2fin],*coeffs_F2) - AnaϕTide2(Xhybarr[numBNS2ref],*coeffs_F2)
        BNS2ϕTideF2arr = np.array(BNS2ϕTideF2).flatten()

        numBNS1off = len(BNS1Tshift)-len(Xhybarr[numBNS1ref:numX1mrg])
        numBNS2off =  len(BNS2Tshift)-len(Xhybarr[numBNS2ref:numX2mrg])
        
        return globals().update(locals())

    def curvefits():
        #now do the curve fit starting from the sets of coeff's from publications (listed in initalize function in this class)
        
        #########FOR BNS1:
        maxcorrectionIDX1=np.where(BNS1ϕTideNR==np.max(BNS1ϕTideNR))[0][0]

        #BNS1, fit only up to merger
        popt, a =curve_fit(AnaϕTide1,Xhybarr[numBNS1ref:numX1mrg], BNS1ϕTideNR[0:-numBNS1off],coeffs_PN)
        pBNS1matchedcoeffsmrg_PN=popt
        BNS1ϕTidemrg_PN = AnaϕTide1(Xhybarr[numBNS1ref:numBNS1fin],*pBNS1matchedcoeffsmrg_PN)
        BNS1ϕTidemrg_PN = BNS1ϕTidemrg_PN - BNS1ϕTidemrg_PN[0]

        popt, a =curve_fit(AnaϕTide1,Xhybarr[numBNS1ref:numX1mrg], BNS1ϕTideNR[0:-numBNS1off],coeffs_F1)
        pBNS1matchedcoeffsmrg_F1=popt
        BNS1ϕTidemrg_F1 = AnaϕTide1(Xhybarr[numBNS1ref:numBNS1fin],*pBNS1matchedcoeffsmrg_F1)
        BNS1ϕTidemrg_F1 = BNS1ϕTidemrg_F1 - BNS1ϕTidemrg_F1[0]

        popt, a =curve_fit(AnaϕTide1,Xhybarr[numBNS1ref:numX1mrg], BNS1ϕTideNR[0:-numBNS1off],coeffs_F2)
        pBNS1matchedcoeffsmrg_F2=popt
        BNS1ϕTidemrg_F2 = AnaϕTide1(Xhybarr[numBNS1ref:numBNS1fin],*pBNS1matchedcoeffsmrg_F2)
        BNS1ϕTidemrg_F2 = BNS1ϕTidemrg_F2 - BNS1ϕTidemrg_F2[0]


        #########FOR BNS2:
        maxcorrectionIDX2=np.where(BNS2ϕTideNR==np.max(BNS2ϕTideNR))[0][0]

        #BNS2, fit only up to merger
        popt, a =curve_fit(AnaϕTide2,Xhybarr[numBNS2ref:numX2mrg], BNS2ϕTideNR[0:-numBNS2off],coeffs_PN)
        pBNS2matchedcoeffsmrg_PN=popt
        BNS2ϕTidemrg_PN = AnaϕTide2(Xhybarr[numBNS2ref:numBNS2fin],*pBNS2matchedcoeffsmrg_PN)
        BNS2ϕTidemrg_PN = BNS2ϕTidemrg_PN - BNS2ϕTidemrg_PN[0]

        popt, a =curve_fit(AnaϕTide2,Xhybarr[numBNS2ref:numX2mrg], BNS2ϕTideNR[0:-numBNS2off],coeffs_F1)
        pBNS2matchedcoeffsmrg_F1=popt
        BNS2ϕTidemrg_F1 = AnaϕTide2(Xhybarr[numBNS2ref:numBNS2fin],*pBNS2matchedcoeffsmrg_F1)
        BNS2ϕTidemrg_F1 = BNS2ϕTidemrg_F1 - BNS2ϕTidemrg_F1[0]

        popt, a =curve_fit(AnaϕTide2,Xhybarr[numBNS2ref:numX2mrg], BNS2ϕTideNR[0:-numBNS2off],coeffs_F2)
        pBNS2matchedcoeffsmrg_F2=popt
        BNS2ϕTidemrg_F2 = AnaϕTide2(Xhybarr[numBNS2ref:numBNS2fin],*pBNS2matchedcoeffsmrg_F2)
        BNS2ϕTidemrg_F2 = BNS2ϕTidemrg_F2 - BNS2ϕTidemrg_F2[0]


        # fit coeffs found for BNS1 to BNS2
        BNS2ϕTide_BNS1F2 = AnaϕTide2(Xhybarr[numBNS2ref:numBNS2fin],*pBNS1matchedcoeffsmrg_F2)
        BNS2ϕTide_BNS1F2 = BNS2ϕTide_BNS1F2 - BNS2ϕTide_BNS1F2[0]
        # fit coeffs found for BNS2 to BNS1
        BNS1ϕTide_BNS2F2 = AnaϕTide1(Xhybarr[numBNS1ref:numBNS1fin],*pBNS2matchedcoeffsmrg_F2)
        BNS1ϕTide_BNS2F2 = BNS1ϕTide_BNS2F2 - BNS1ϕTide_BNS2F2[0]

        return globals().update(locals())
        
    def retrievequantities():
        b=globals()
        return b

"""-------------------------------------------"""
class Tapering():
    def initialize(globalvars):
        globals().update(globalvars)
        
        #the analytic phase arrays for BBH and BNS1
        #MϕBBH1hybarr = Mϕhybarr[numBNS1ref:numBNS1fin]-Mϕhybarr[numBNS1ref]
        MϕBBH1hybarr = AnaBBH1ϕintarr - AnaBBH1ϕintarr[0]
        MϕBNS1hybarr = MϕBBH1hybarr + (BNS1ϕTidemrg_F2)
        MϕBNS1NRarr = BNS1ϕintarr-BNS1ϕintarr[0]
        
        #the analytic phase arrays for BBH and BNS2
        #MϕBBH2hybarr = Mϕhybarr[numBNS2ref:numBNS2fin]-Mϕhybarr[numBNS2ref]
        MϕBBH2hybarr = AnaBBH2ϕintarr - AnaBBH2ϕintarr[0]
        MϕBNS2hybarr = MϕBBH2hybarr + (BNS2ϕTidemrg_F2)
        MϕBNS2NRarr = BNS2ϕintarr-BNS2ϕintarr[0]

        #too noisy to be of any use - attempt to taper
        MϕBNS1diff=MϕBNS1hybarr-MϕBNS1NRarr

        MϕBNS1diff_dot=np.gradient(MϕBNS1diff,BNS1Tshift) 

        #too noisy to be of any use - attempt to taper
        MϕBNS2diff=MϕBNS2hybarr-MϕBNS2NRarr
        
        MϕBNS2diff_dot=np.gradient(MϕBNS2diff,BNS2Tshift)

        return globals().update(locals())       

    def taper():

        #outputs where derivative of difference wrt time goes from negative to positive...possible cutoff points (want to find element after the "small hump around t=3000")
        cutoffs1=[]
        for i in range(len(MϕBNS1diff_dot)-1):
            if MϕBNS1diff_dot[i+1]*MϕBNS1diff_dot[i]<=0:
                cutoffs1.append(i)
        
        #outputs where derivative of difference wrt time goes from negative to positive...possible cutoff points (want to find element after the "small hump around t=3000")
        cutoffs2=[]
        for i in range(len(MϕBNS2diff_dot)-1):
            if MϕBNS2diff_dot[i+1]*MϕBNS2diff_dot[i]<=0:
                cutoffs2.append(i)

        #generating all "corrected/tapered phase-candidates" at once and making/appending to a list 
        #where each element in the list is one of these "phase arrays"
        Listofphasearrays1=[]
        for i in range(len(cutoffs1)):
            MϕBNS1Tidal=np.zeros(len(BNS1Tshift))
            for idx in range(0,cutoffs1[i]):
                MϕBNS1Tidal[idx]=BNS1ϕTidemrg_F2[idx]+AnaBBH1ϕintarr[idx]-(BNS1ϕTidemrg_F2[0]+AnaBBH1ϕintarr[0])
            for idx in range(cutoffs1[i],len(BNS1Tshift)):
                MϕBNS1Tidal[idx]=AnaBBH1ϕintarr[idx]+(BNS1ϕTidemrg_F2[cutoffs1[i]-1])-(BNS1ϕTidemrg_F2[0]+AnaBBH1ϕintarr[0])    
            Listofphasearrays1.append( MϕBNS1Tidal)

        Listofphasearrays2=[]
        for i in range(len(cutoffs2)):
            MϕBNS2Tidal=np.zeros(len(BNS2Tshift))
            for idx in range(0,cutoffs2[i]):
                MϕBNS2Tidal[idx]=BNS2ϕTidemrg_F2[idx]+AnaBBH2ϕintarr[idx]-(BNS2ϕTidemrg_F2[0]+AnaBBH2ϕintarr[0])
            for idx in range(cutoffs2[i],len(BNS2Tshift)):
                MϕBNS2Tidal[idx]=AnaBBH2ϕintarr[idx]+(BNS2ϕTidemrg_F2[cutoffs2[i]-1])-(BNS2ϕTidemrg_F2[0]+AnaBBH2ϕintarr[0])    
            Listofphasearrays2.append(MϕBNS2Tidal)

        MϕBNS1hybarr_hvs=Listofphasearrays1[9]
        MϕBNS2hybarr_hvs=Listofphasearrays2[21]
        return globals().update(locals())
        
    def retrievequantities():
        b=globals()
        return b
"""--------------------------------------------"""

class TidalAmp():
    def initialize(globalvars):
        globals().update(globalvars)
        # now we can take the differences between the Amplitudes to find the tidal correction

        BNS1AmpTideNR = ((BNS1Ampintarr/BNS1Ampintarr[0])/(BBH1Ampintarr/BBH1Ampintarr[0])) - 1.
        BNS2AmpTideNR = ((BNS2Ampintarr/BNS2Ampintarr[0])/(BBH2Ampintarr/BBH2Ampintarr[0])) - 1.

        Tcutoff1 = cutoffs1[9]
        Tcutoff2 = cutoffs2[21]

        ###FOR BNS1:
        # the analytical approximation for merger amplitude
        kTeff1 = (3./32.)*(BNS1tΛ+BNS1tΛ)
        AnaAmrg1 = η*1.6498*(1.+((2.5603e-2)*(kTeff1))-((1.024e-5)*((kTeff1)**2))) \
                /(1+(4.7278e-2)*(kTeff1))
        # the analytical approximation for merger X parameter
        AnaMΩmrg1 = 0.1793*(q)**(1/2)*(1.+ (3.354e-2)*kTeff1 + (4.315e-5)*kTeff1**2) \
                /(1.+ (7.542e-2)*kTeff1 + (2.236e-4)*kTeff1**2)
        AnaXmrg1 = (AnaMΩmrg1)**(2/3)
        # consider Xmrg when the stars are touching, making first contact
        Xc1 = 2.8/RT 

        # the numerical values for merger amplitude
        Xmerge1 = Xhybarr[numBNS1ref+mrgBNS1index]
        fact1 = BBH1Ampintarr[0]/BNS1Ampintarr[0]
        Amerge1 = BNS1Ampintarr[mrgBNS1index]*fact1

        #computing "initial value"/"initial guess" for free parameter d
        p1 = 1 #power of XpN
        dA1=(1./Xmerge1)*((AnaAmpTide1(Xmerge1,0, p1)/Amerge1)-1.)
        dAnaA1=(1./AnaXmrg1)*((AnaAmpTide1(AnaXmrg1,0, p1)/AnaAmrg1)-1.)
#        dAnaA1=(1./Xc1)*((AnaAmpTide1(Xc1,0, p1)/Xc1)-1.)

        # PN fit
        initialPNAcoeffs1 = np.array([0,p1])
        # numerical fit
        initialAnaAcoeffs1 = np.array([dAnaA1,p1])
        initialAcoeffs1 = np.array([dA1,p1])
        
        ###FOR BNS 2:
        # the analytical approximation for merger amplitude
        kTeff2 = (3./32.)*(BNS2tΛ+BNS2tΛ)
        AnaAmrg2 = η*1.6498*(1.+((2.5603e-2)*(kTeff2))-((1.024e-5)*((kTeff2)**2))) \
                /(1+(4.7278e-2)*(kTeff2))
        # the analytical approximation for merger X parameter
        AnaMΩmrg2 = 0.1793*(q)**(1/2)*(1.+ (3.354e-2)*kTeff2 + (4.315e-5)*kTeff2**2) \
                /(1.+ (7.542e-2)*kTeff2 + (2.236e-4)*kTeff2**2)
        AnaXmrg2 = (AnaMΩmrg2)**(2/3)
        # consider Xmrg when the stars are touching, making first contact
        Xc2 = 2.7/RT 

        # the numerical values for merger amplitude
        Xmerge2 = Xhybarr[numBNS2ref+mrgBNS2index]
        fact2 = BBH2Ampintarr[0]/BNS2Ampintarr[0]
        Amerge2=BNS2Ampintarr[mrgBNS2index]*fact2

        #computing "initial value"/"initial guess" for free parameter d
        p2 = 1 #power of XpN
        dA2=(1./Xmerge2)*((AnaAmpTide2(Xmerge2,0, p2)/Amerge2)-1.)
        dAnaA2=(1./AnaXmrg2)*((AnaAmpTide2(AnaXmrg2,0, p2)/AnaAmrg2)-1.)
#        dAnaA2=(1./Xc2)*((AnaAmpTide2(Xc2,0, p2)/Xc2)-1.)
        
        # PN fit
        initialPNAcoeffs2 = np.array([0,p2])
        # numerical fit
        initialAnaAcoeffs2 = np.array([dAnaA2,p2])
        initialAcoeffs2 = np.array([dA2,p2])

        return globals().update(locals())

    def AnaApprox():
        # PN amplitude BNS1, for numerical and analytic merger
        BNS1PNATide = AnaAmpTide1(Xhybarr[numBNS1ref:numBNS1ref+Tcutoff1],*initialPNAcoeffs1)-AnaAmpTide1(Xhybarr[numBNS1ref],*initialPNAcoeffs1)
        BNS1PNATidearr = np.array(BNS1PNATide).flatten()

        # PN amplitude BNS2, for numerical and analytic merger
        BNS2PNATide = AnaAmpTide2(Xhybarr[numBNS2ref:numBNS2ref+Tcutoff2],*initialPNAcoeffs2)-AnaAmpTide2(Xhybarr[numBNS2ref],*initialPNAcoeffs2)
        BNS2PNATidearr = np.array(BNS2PNATide).flatten()

        # numerical fit amplitude, for numerical and analytic merger
        BNS1ATide = AnaAmpTide1(Xhybarr[numBNS1ref:numBNS1ref+Tcutoff1],*initialAcoeffs1)-AnaAmpTide1(Xhybarr[numBNS1ref],*initialAcoeffs1)
        BNS1ATidearr = np.array(BNS1ATide).flatten()
        BNS1AnaATide = AnaAmpTide1(Xhybarr[numBNS1ref:numBNS1ref+Tcutoff1],*initialAnaAcoeffs1)-AnaAmpTide1(Xhybarr[numBNS1ref],*initialAnaAcoeffs1)
        BNS1AnaATidearr = np.array(BNS1AnaATide).flatten()

        BNS2ATide = AnaAmpTide2(Xhybarr[numBNS2ref:numBNS2ref+Tcutoff2],*initialAcoeffs2)-AnaAmpTide2(Xhybarr[numBNS2ref],*initialAcoeffs2)
        BNS2ATidearr = np.array(BNS2ATide).flatten()
        BNS2AnaATide = AnaAmpTide2(Xhybarr[numBNS2ref:numBNS2ref+Tcutoff2],*initialAnaAcoeffs2)-AnaAmpTide2(Xhybarr[numBNS2ref],*initialAnaAcoeffs2)
        BNS2AnaATidearr = np.array(BNS2AnaATide).flatten()
        return globals().update(locals())

    def curvefits():
        ###FOR BNS1:
        #PN fitting cannot reach cutoff, must stop at merger mrgBNS1index
        popt, a =curve_fit(AnaAmpTide1,Xhybarr[numBNS1ref:numBNS1ref+mrgBNS1index],(BNS1AmpTideNR[0:mrgBNS1index]),initialPNAcoeffs1,maxfev=150000)
        matchedPNAcoeffs1=popt
        BNS1PNATidefit = AnaAmpTide1(Xhybarr[numBNS1ref:numBNS1ref+Tcutoff1],*matchedPNAcoeffs1)
        BNS1PNATidefit = BNS1PNATidefit - BNS1PNATidefit[0]

        #numeric fit fitting up to cutoff
        popt, a =curve_fit(AnaAmpTide1,Xhybarr[numBNS1ref:numBNS1ref+mrgBNS1index],(BNS1AmpTideNR[0:mrgBNS1index]),initialAcoeffs1,maxfev=150000)
        matchedAcoeffs1=popt
        BNS1ATidefit = AnaAmpTide1(Xhybarr[numBNS1ref:numBNS1ref+Tcutoff1],*matchedAcoeffs1)
        BNS1ATidefit = BNS1ATidefit - BNS1ATidefit[0]

        popt, a =curve_fit(AnaAmpTide1,Xhybarr[numBNS1ref:numBNS1ref+mrgBNS1index],(BNS1AmpTideNR[0:mrgBNS1index]),initialAnaAcoeffs1,maxfev=150000)
        matchedAnaAcoeffs1=popt
        BNS1AnaATidefit = AnaAmpTide1(Xhybarr[numBNS1ref:numBNS1ref+Tcutoff1],*matchedAnaAcoeffs1)
        BNS1AnaATidefit = BNS1AnaATidefit - BNS1AnaATidefit[0]

        #attempt to cutoff the amplitude at merger
        AmpBNS1premrg=np.zeros(Tcutoff1)
        p=0
        for i in BNS1ATidefit[0:Tcutoff1]:
            AmpBNS1premrg[p]=(1+i)*AnaBBH1Aintarr[p]/AnaBBH1Aintarr[0]
            p=p+1

        A1_min = 0
        A1_max = AmpBNS1premrg[mrgBNS1index]
        Aclip1 = np.clip(AmpBNS1premrg[mrgBNS1index:Tcutoff1], A1_min,A1_max)

        # could we apply a hanning tapering window?
        AmpBNS1pm=np.zeros(mrgBNS1index)
        p=0
        for i in BNS1ATidefit[0:mrgBNS1index]:
            AmpBNS1pm[p]=(1.+i)*AnaBBH1Aintarr[p]
            p=p+1

        AmpBNS1han = AmpBNS1pm[-1]*np.hanning(2*(Tcutoff1-(mrgBNS1index+1)))
        halfPoint1 = np.argmax(AmpBNS1han)
        AmpBNS1han = AmpBNS1han[halfPoint1:]


        ###For BNS2:
        #PN fitting up to merger
        popt, a =curve_fit(AnaAmpTide2,Xhybarr[numBNS2ref:numBNS2ref+mrgBNS2index],(BNS2AmpTideNR[0:mrgBNS2index]),initialPNAcoeffs2,maxfev=150000)
        matchedPNAcoeffs2=popt
        BNS2PNATidefit = AnaAmpTide2(Xhybarr[numBNS2ref:numBNS2ref+Tcutoff2],*matchedPNAcoeffs2)
        BNS2PNATidefit = BNS2PNATidefit -  BNS2PNATidefit[0]

        #numeric fit fitting up to merger
        popt, a =curve_fit(AnaAmpTide2,Xhybarr[numBNS2ref:numBNS2ref+mrgBNS2index],(BNS2AmpTideNR[0:mrgBNS2index]),initialAcoeffs2,maxfev=150000)
        matchedAcoeffs2=popt
        BNS2ATidefit = AnaAmpTide2(Xhybarr[numBNS2ref:numBNS2ref+Tcutoff2],*matchedAcoeffs2)
        BNS2ATidefit = BNS2ATidefit -  BNS2ATidefit[0]

        popt, a =curve_fit(AnaAmpTide2,Xhybarr[numBNS2ref:numBNS2ref+mrgBNS2index],(BNS2AmpTideNR[0:mrgBNS2index]),initialAnaAcoeffs2,maxfev=150000)
        matchedAnaAcoeffs2=popt
        BNS2AnaATidefit = AnaAmpTide2(Xhybarr[numBNS2ref:numBNS2ref+Tcutoff2],*matchedAnaAcoeffs2)
        BNS2AnaATidefit = BNS2AnaATidefit -  BNS2AnaATidefit[0]

        #attempt to cutoff the amplitude at merger
        # numeric fit
        AmpBNS2premrg=np.zeros(Tcutoff2)
        p=0
        for i in BNS2ATidefit[0:Tcutoff2]:
            AmpBNS2premrg[p]=(1.+i)*AnaBBH2Aintarr[p]/AnaBBH2Aintarr[0]
            p=p+1

        A2_min = 0
        A2_max = AmpBNS2premrg[mrgBNS2index]
        Aclip2 = np.clip(AmpBNS2premrg[mrgBNS2index:Tcutoff2], A2_min,A2_max)

        # could we apply a hanning tapering window?
        AmpBNS2pm=np.zeros(mrgBNS2index)
        p=0
        for i in BNS2ATidefit[0:mrgBNS2index]:
            AmpBNS2pm[p]=(1.+i)*AnaBBH2Aintarr[p]
            p=p+1

        AmpBNS2han = AmpBNS2pm[-1]*np.hanning(2*(Tcutoff2-(mrgBNS2index+1)))
        halfPoint2 = np.argmax(AmpBNS2han)
        AmpBNS2han = AmpBNS2han[halfPoint2:]

        return globals().update(locals())

    def retrievequantities():
        b=globals()
        return b
        
class ReconstructVars():
    # reconstruct all the variables up to the cutoff time
    def initialize(globalvars):
        globals().update(globalvars)
        Tcutoff1 = cutoffs1[9]
        Tcutoff2 = cutoffs2[21]
        # the new time arrays
        BNS1Tnew = BNS1Tshift[0:Tcutoff1]
        BNS2Tnew = BNS2Tshift[0:Tcutoff2]
        # reconstructed Mϕ for BNS1/NR
        MϕBNS1new = MϕBNS1hybarr_hvs[0:Tcutoff1] - MϕBNS1hybarr_hvs[0]
        MϕBNS1NRnew = MϕBNS1NRarr[0:Tcutoff1] - MϕBNS1NRarr[0]
        # reconstructed Mϕ for BNS2/NR
        iΔTrefBNS12 = int(ΔTrefBNS12/dt) 
        ΔϕBNS2BBH = AnaBBH1ϕintarr[iΔTrefBNS12]-AnaBBH1ϕintarr[0]
        MϕBNS2new = MϕBNS2hybarr_hvs[0:Tcutoff2]-MϕBNS2hybarr_hvs[0]+ΔϕBNS2BBH
        MϕBNS2NRnew = MϕBNS2NRarr[0:Tcutoff2] - MϕBNS2NRarr[0]+ΔϕBNS2BBH
        ##calculate the angular frequencies for BNS1
        MωBNS1new = np.gradient(MϕBNS1new, BNS1Tnew)
        MωBNS1NRnew  = np.gradient(MϕBNS1NRnew, BNS1Tnew)
        ##calculate the angular frequencies for BNS2
        MωBNS2new = np.gradient(MϕBNS2new, BNS2Tnew)
        MωBNS2NRnew  = np.gradient(MϕBNS2NRnew, BNS2Tnew)
        # calculate the PN parameter for BNS1
        XBNS1new = np.sign(MωBNS1new/2.)*(np.abs(MωBNS1new/2.))**(2/3)
        XBNS1NRnew  = np.sign(MωBNS1NRnew/2.)*(np.abs(MωBNS1NRnew/2.))**(2/3)
        # calculate the PN parameter for BNS2
        XBNS2new = np.sign(MωBNS2new/2.)*(np.abs(MωBNS2new/2.))**(2/3)
        XBNS2NRnew  = np.sign(MωBNS2NRnew/2.)*(np.abs(MωBNS2NRnew/2.))**(2/3)
        #calculate the orbital velocity for BNS1
        VBNS1new = np.sign(XBNS1new)*(np.abs(XBNS1new))**(1/2)
        VBNS1NRnew  = np.sign(XBNS1NRnew)*(np.abs(XBNS1NRnew))**(1/2)
        #calculate the orbital velocity for BNS2
        VBNS2new = np.sign(XBNS2new)*(np.abs(XBNS2new))**(1/2)
        VBNS2NRnew  = np.sign(XBNS2NRnew)*(np.abs(XBNS2NRnew))**(1/2)
        # calculate the Orbital Sep for BNS1
        RBNS1new = np.array(r_PN(XBNS1new)).flatten()
        RBNS1NRnew  = np.array(r_PN(XBNS1NRnew)).flatten()
        # calculate the Orbital Sep for BNS2
        RBNS2new = np.array(r_PN(XBNS2new)).flatten()
        RBNS2NRnew  = np.array(r_PN(XBNS2NRnew)).flatten()
        #calculate the radial velocity for BNS1
        VRBNS1new = np.gradient(RBNS1new, BNS1Tnew)
        VRBNS1NRnew  = np.gradient(RBNS1NRnew, BNS1Tnew)
        #calculate the radial velocity for BNS2
        VRBNS2new = np.gradient(RBNS2new, BNS2Tnew)
        VRBNS2NRnew  = np.gradient(RBNS2NRnew, BNS2Tnew)
        #position of first star in BNS1/BBH:
        rABNS1new = (mB/m)*RBNS1new 
        rABNS1NRnew  = (mB/m)*RBNS1NRnew 
        #orbital velocity of first star BNS1/BBHs
        vABNS1new = (mB/m)*VBNS1new 
        vABNS1NRnew  = (mB/m)*VBNS1NRnew 
        #radial velocity of first star BNS1/BBHs
        vARBNS1new = (mB/m)*VRBNS1new 
        vARBNS1NRnew  = (mB/m)*VRBNS1NRnew 
        #position of first star in BNS2/BBH:
        rABNS2new = (mB/m)*RBNS2new 
        rABNS2NRnew  = (mB/m)*RBNS2NRnew 
        #orbital velocity of first star BNS2/BBHs
        vABNS2new = (mB/m)*VBNS2new 
        vABNS2NRnew  = (mB/m)*VBNS2NRnew 
        #radial velocity of first star BNS2/BBHs
        vARBNS2new = (mB/m)*VRBNS2new 
        vARBNS2NRnew  = (mB/m)*VRBNS2NRnew 
        # reconstructed Amp and Mϕ BNS1/BNS2
        AmpBNS1new = np.concatenate((AmpBNS1pm[0:mrgBNS1index], AmpBNS1han))
        AmpBNS2new = np.concatenate((AmpBNS2pm[0:mrgBNS2index], AmpBNS2han))
        AmpBNS1NRnew = fact1*BNS1Ampintarr[0:Tcutoff1] 
        AmpBNS2NRnew = fact2*BNS2Ampintarr[0:Tcutoff2]

        # calculate all reconstructed quantities for hybrid BBH
        MωBBH1hybarr = np.gradient(MϕBBH1hybarr, BNS1Tshift)
        MωBBH2hybarr = np.gradient(MϕBBH2hybarr, BNS2Tshift)
        XBBH1hybarr = np.sign(MωBBH1hybarr/2.)*(np.abs(MωBBH1hybarr/2.))**(2/3)
        XBBH2hybarr = np.sign(MωBBH2hybarr/2.)*(np.abs(MωBBH2hybarr/2.))**(2/3)
        VBBH1hybarr = np.sign(XBBH1hybarr)*(np.abs(XBBH1hybarr))**(1/2)
        VBBH2hybarr = np.sign(XBBH2hybarr)*(np.abs(XBBH2hybarr))**(1/2)
        RBBH1hybarr = np.array(r_PN(XBBH1hybarr)).flatten()
        RBBH2hybarr = np.array(r_PN(XBBH2hybarr)).flatten()
        VRBBH1hybarr = np.gradient(RBBH1hybarr, BNS1Tshift)
        VRBBH2hybarr = np.gradient(RBBH2hybarr, BNS2Tshift)
        rABBH1hybarr = (mB/m)*RBBH1hybarr 
        rABBH2hybarr = (mB/m)*RBBH2hybarr 
        vABBH1hybarr = (mB/m)*VBBH1hybarr 
        vABBH2hybarr = (mB/m)*VBBH2hybarr 
        vARBBH1hybarr = (mB/m)*VRBBH1hybarr 
        vARBBH2hybarr = (mB/m)*VRBBH2hybarr 

        return globals().update(locals())

    def retrievequantities():
        b=globals()
        return b
