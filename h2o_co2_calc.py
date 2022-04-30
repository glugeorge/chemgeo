# Import relevant libraries
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Function for equlibrium pressures
def compute_equilib_pressure(input_comp,temperature,h2o,co2):
    
    # Constants
    d_h2o = -16.4
    d_al = 4.4
    d_feo_mgo = -17.1
    d_na2o_k2o = 22.8
    
    a_co2 = 1
    b_co2 = 17.3
    C_co2 = 0.12
    B_co2 = -6
    
    a_h2o = 0.53
    b_h2o = 2.35
    C_h2o = 0.02
    B_h2o = -3.37
    
    # component molecular masses for [al2o3,cao,feo,k2o,mgo,na2o,sio2,tio2]
    # get molecular masses to normalize
    comp_mol_mass = [101.96,56.08,71.85,94.20,40.32,61.98,60.09,79.90]
    frac_input_comp = input_comp/(np.sum(input_comp)/100)
    norm_comp = frac_input_comp/comp_mol_mass
    h2o_raw = h2o
    h2o_norm = h2o/18.01
    
    total = np.sum(norm_comp)+h2o_norm
    
    # get composition used for calculation
    al2o3 = norm_comp[0] / total
    cao = norm_comp[1] / total
    feo = norm_comp[2] / total
    k2o = norm_comp[3] / total
    mgo = norm_comp[4] / total
    na2o = norm_comp[5] / total
    sio2 = norm_comp[6] / total
    tio2 = norm_comp[7] / total
    h2o = h2o_norm / total
    
    # calculate nbo
    nbo = 2 * (h2o+k2o+na2o+cao+mgo+feo-al2o3)
    nbo_o = nbo/(2*sio2+2*tio2+3*al2o3+mgo+feo+cao+na2o+k2o+h2o)
    
    # setup equations - 3 equations, 3 unknowns (partial p for h2o and co2, total p)
    def equations(x):
        # ph2o,pco2,p = x[0],x[1],x[2]
        T = temperature+273.15
        return([
            # equation 12
            ((h2o*d_h2o + al2o3*d_al/(cao+k2o+na2o) + (feo+mgo)*d_feo_mgo +
             (na2o+k2o)*d_na2o_k2o) + a_co2*np.log(x[1]) + b_co2*nbo_o+B_co2 + 
             C_co2*x[2]/T) - np.log(co2),
            # equation 13
            a_h2o*np.log(x[0]) + b_h2o*nbo_o + B_h2o + C_h2o*x[2]/T - np.log(h2o_raw),
            # Partial pressures sum
            x[0]+x[1]-x[2]
            ])
    # Solve equations for unknowns
    solution = fsolve(equations,[1,1,1])

    # Return ph2o,pco2,p = x[0],x[1],x[2]
    return solution

# Function for calculating contents
def compute_contents(input_comp,temperature,start_p,mol_frac_co2):
    end_p = 1 # in bar = 0.1 MPa
    
    # Order array of pressures from initial P to 0.1 MPa, though initial P should always be greater
    if start_p<end_p:
        pressures = np.linspace(start_p,end_p)
    else:
        pressures = np.linspace(end_p,start_p)
    
    # Constants
    d_h2o = -16.4
    d_al = 4.4
    d_feo_mgo = -17.1
    d_na2o_k2o = 22.8
    
    a_co2 = 1
    b_co2 = 17.3
    C_co2 = 0.12
    B_co2 = -6
    
    a_h2o = 0.53
    b_h2o = 2.35
    C_h2o = 0.02
    B_h2o = -3.37
    
    # component molecular masses for [al2o3,cao,feo,k2o,mgo,na2o,sio2,tio2]
    # normalize by molecular mass
    comp_mol_mass = [101.96,56.08,71.85,94.20,40.32,61.98,60.09,79.90]
    frac_input_comp = input_comp/(np.sum(input_comp)/100)
    norm_comp = frac_input_comp/comp_mol_mass
    
    # Initialize arrays to hold calculated solubilities
    h2o_conts = np.zeros(np.shape(pressures))
    co2_conts = np.zeros(np.shape(pressures))
    i = 0
    # iterate through each pressure
    for P in pressures:
        def equations(x):
            # h2o content (ppm), co2 content = x[0],x[1]
            T = temperature+273.15
            h2o = x[0]
            h2o_raw = h2o
            h2o_norm = h2o/18.01 # normalize by molecular mass
            
            total = np.sum(norm_comp)+h2o_norm
            
            # get normalized composition used for calculation
            al2o3 = norm_comp[0] / total
            cao = norm_comp[1] / total
            feo = norm_comp[2] / total
            k2o = norm_comp[3] / total
            mgo = norm_comp[4] / total
            na2o = norm_comp[5] / total
            sio2 = norm_comp[6] / total
            tio2 = norm_comp[7] / total
            h2o = h2o_norm / total
            
            # calculate nbo
            nbo = 2 * (h2o+k2o+na2o+cao+mgo+feo-al2o3)
            nbo_o = nbo/(2*sio2+2*tio2+3*al2o3+mgo+feo+cao+na2o+k2o+h2o)
            
            # Define equations to solve
            return([
                # Equation 13
                a_h2o*np.log(P*(1-mol_frac_co2)) + b_h2o*nbo_o + B_h2o + C_h2o*P/T - np.log(h2o_raw),
                
                # Equation 12
                ((h2o*d_h2o + al2o3*d_al/(cao+k2o+na2o) + (feo+mgo)*d_feo_mgo +
                  (na2o+k2o)*d_na2o_k2o) + a_co2*np.log(P*(mol_frac_co2)) + b_co2*nbo_o+B_co2 + 
                 C_co2*P/T) - np.log(x[1]),
                ])
        solution = fsolve(equations,[0.01,1])
        # add solution to arrays
        h2o_conts[i] = solution[0]
        co2_conts[i] = solution[1]
        i += 1
    # return array of solubilities
    return h2o_conts, co2_conts

# main organizational code
if __name__ == "__main__":
    # Comment out for testing
    # What calculation
    compute_opt = int(input("Calculation: \n \
                            0 - Equilibrium Pressure \n \
                            1 - CO2 and H2O contents \n") or "0") # 0 = equilib pressure, 1 = contents from P to 0.1MPa
    if compute_opt != 0 and compute_opt != 1:
        print("Invalid entry, run program again")
    
    else:
        # Change these compositions
        # Default based of default settings on web-app
        al2o3 = float(input("Al2O3 content (wt): ") or "14.32")
        cao = float(input("CaO content (wt): ") or "12.25")
        feo = float(input("FeO content (wt): ") or "11.24")
        k2o = float(input("K2O content (wt): ") or "2")
        mgo = float(input("MgO content (wt): ") or "7.11")
        na2o = float(input("Na2O content (wt): ") or "2.92")
        sio2 = float(input("SiO2 content (wt): ") or "49.46")
        tio2 = float(input("TiO2 content (wt): ") or "1.58")
        input_comp = [al2o3,cao,feo,k2o,mgo,na2o,sio2,tio2]
        temperature = float(input("Temperature (C): ") or "1200")
        if compute_opt == 0: # equilib. pressure
            
            h2o = float(input("H2O content (wt): ") or "1")
            co2 = float(input("CO2 content (ppm): ") or "2000")
            print("\nCalculating equilibrium pressure...\n")
            sol = compute_equilib_pressure(input_comp,temperature,h2o,co2)
            print(f"Equilibrium pressure is {sol[2]} bar.")
            
        else: # contents
            start_p = float(input("Starting pressure (bar): ") or "1000")
            mol_frac_co2 = float(input("Carbon dioxide gas mole fraction: ") or "0.5")
            print("\nCalculating contents...\n")
            h2o_conts,co2_conts = compute_contents(input_comp,temperature,start_p,mol_frac_co2)
            # return solutions at initial pressure
            print(f"Contents at initial pressure:\n \
            h2o: {h2o_conts[-1]} wt\n \
            co2: {co2_conts[-1]} ppm")
            
            # plot solutions
            pressures = np.linspace(1,start_p)[1:]
            fig, ax1 = plt.subplots()
            color = 'tab:blue'
            ax1.scatter(pressures,h2o_conts[1:],color=color)
            ax1.set_xlabel("Pressures (bar)")
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.set_ylabel("H2O content (wt)",color=color)
            ax1.invert_xaxis()
            ax1.set_title("H2O-CO2 contents")
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.scatter(pressures,co2_conts[1:],color=color)
            ax2.set_ylabel("CO2 content (ppm)",color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            plt.show()

    
    # This section is to generate plots for testing, commented out for actual program
    # Recreating figures from Iacono-Marziano et al., (2012)
    # Mount Etna compositions: from Table 1
    # [al2o3,cao,feo,k2o,mgo,na2o,sio2,tio2]
'''
    input_comp = [17.32,10.93,10.24,1.99,5.76,3.45,47.95,1.67]
    
    # Values from h2o-co2 experiments in Table 2
    h2o_measured = [0.95,1.18,0.99,2.12,0.80,2.08,1.43,3.01,1.04,2.82,0.87,3.45,2.70,1.64,1.09,3.54,1.36,1.47,3.31,2.16]
    
    co2_measured = [306,191,843,548,808,534,1278,1035,1853,1498,1706,1408,1412,2515,2816,2416,3673,3965,4061,4230]
    
    pressures_measured = [485,485,1015,1015,1017,1017,1530,1530,2047,2047,2055,2055,2135,2754,3080,3080,4185,4185,4185,4185]
    
    co2_mol_fracs = [0.68,0.56,0.84,0.49,0.88,0.51,0.81,0.43,0.90,0.61,0.93,0.48,0.65,0.87,0.94,0.64,0.94,0.93,0.76,0.87]
    
    temperature = 1200
    
    # Testing Equlibrium pressure
    pressures_calculated = []

    for i in range(len(h2o_measured)):
        equilib_pressure = compute_equilib_pressure(input_comp,temperature,h2o_measured[i],co2_measured[i])[2]
        pressures_calculated.append(equilib_pressure)
    plt.figure()
    plt.scatter(pressures_measured,pressures_calculated)
    plt.plot([0,4500],[0,4500])
    plt.xlabel("Iacono-Marziano et al. Experiment Pressures (bar)")
    plt.ylabel("Calculated equilibrium pressures (bar)")
    plt.title("Validating calculated pressures using data from Iacono-Marziano et al. Table 2")
    plt.show()
    # Testing contents
    co2_comp_calculated = []
    h2o_comp_calculated = []
    for i in range(len(h2o_measured)):
        h2o_conts,co2_conts = compute_contents(input_comp,temperature,pressures_measured[i],co2_mol_fracs[i])
        co2_comp_calculated.append(co2_conts[-1])
        h2o_comp_calculated.append(h2o_conts[-1])
    plt.figure()
    plt.scatter(co2_measured,co2_comp_calculated)
    plt.plot([0,5000],[0,5000])
    plt.xlabel("Iacono-Marziano et al. Experiment CO2 compositions (ppm)")
    plt.ylabel("Calculated CO2 compositions (ppm)")
    plt.title("Validating calculated CO2 compositions using data from Iacono-Marziano et al. Table 2")
    plt.show()
    plt.figure()
    plt.scatter(h2o_measured,h2o_comp_calculated)
    plt.plot([0,4.500],[0,4.500])
    plt.xlabel("Iacono-Marziano et al. Experiment H2O compositions (wt)")
    plt.ylabel("Calculated H2O compositions (wt)")
    plt.title("Validating calculated H2O compositions using data from Iacono-Marziano et al. Table 2")
    plt.show()
'''
