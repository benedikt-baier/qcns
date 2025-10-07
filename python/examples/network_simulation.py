import sys
sys.path.append("../../../")
import qcns

def main():
    
    sim = qcns.Simulation()
    
    host_1 = qcns.Host(0, sim)
    host_2 = qcns.Host(1, sim)
    
    host_1.set_eqs_connection(host_2)

if __name__ == '__main__':
    main()