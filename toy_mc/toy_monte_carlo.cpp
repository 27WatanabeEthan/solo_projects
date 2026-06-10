#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <vector>
#include <iomanip>

class Reactor; // we have to initialize the Reactor class so that the neutron class can reference its properties
class Section{
    private:
        double Sig_s, Sig_a, Sig_f, Sig_t;
        double size;
    public:
        double get_size() const { return size; }
        double get_Sig_s() const { return Sig_s; }
        double get_Sig_a() const { return Sig_a; }
        double get_Sig_f() const { return Sig_f; }
        double get_Sig_t() const { return Sig_t; }
    Section(double Sig_s, double Sig_a, double Sig_f, double size){
        this->Sig_s = Sig_s;
        this->Sig_a = Sig_a;
        this->Sig_f = Sig_f;
        this->size = size;
        this->Sig_t = Sig_a + Sig_s + Sig_f;
    }
};
class neutron{
    private:
        // double distance;
        double x = 0;
        double y = 0;
        double z = 0;
        double r = 0;

    public:
        double get_r() const { return r; }

        // these functions will be defined after the Reactor class has been fully defined
        // pass by reference is faster
        double sample_dist(Reactor &reactor, const Section &current_section);
        void traverse(Reactor &reactor);
        void initialize(Reactor &reactor, int core_section_num);
        int sample_collision(Reactor &reactor, const Section &current_section);

        void print_coords(){
            std::cout << "r = " << r << ": ";
            std::cout << "(" << x << ", " << y << ", " << z << ")\n";
        }
};
class Reactor{
    private:
        double size = 0;
        int geometry = 0;
        /*
        0 -> slab
        1 -> sphere
        */
        int sections = 0;
        
    public:
    Reactor(int _geometry){ geometry = _geometry; }
        std::vector<Section> setup;
        neutron n0;
        std::random_device rd;
        int transmitted = 0;
        int absorbed = 0;
        int fissioned = 0;
        int reflected = 0;

        double get_size() const { return size; }
        int get_geometry() const { return geometry; }
        int get_sections() const { return sections; }
        void add_section(const Section &section){
            // sections must be added in order that you wish for them to appear
            size += section.get_size();
            setup.push_back(section);
            sections++;
        }
        int get_location(double dist) const {
            double coord = setup[0].get_size(); // get the coord of the first boundary
            for(size_t i = 0; i < setup.size(); i++){
                // check if we are within the boundary
                if(dist > size){
                    // we have transmitted the reactor
                    return -1;
                }else if(0 <= dist && dist < coord){
                    return i;
                }else if(dist < 0){
                    // we have reflected out of the reactor
                    return -1;
                }
                else{
                    coord += setup[i+1].get_size();
                }
            }
            return -1;
        }
        void reset_tallies(){
            transmitted = 0;
            absorbed = 0;
            fissioned = 0;
        }

};

// now these neutron methods can be implemented after Reactor class has been fully defined
double neutron::sample_dist(Reactor &reactor, const Section &current_section){
    std::uniform_real_distribution<double> random{0.0, 1.0};
    double s = -log(1 - random(reactor.rd)) / current_section.get_Sig_t();
    return s;
}
void neutron::traverse(Reactor &reactor){
    std::uniform_real_distribution<double> random{0.0, 1.0};

    bool is_alive = true;
    if(reactor.get_geometry() == 1){
        // spherical reactor
        while(is_alive){
            int current_section_num = reactor.get_location(r);
            Section current_section = reactor.setup[current_section_num];
            // to determine if the neutron has crossed a boundary or not
            double ri = 0;
            for(int i = 0; i < current_section_num; i++){
                ri += reactor.setup[i].get_size();
                // ro += reactor.setup[i].get_size();
            }
            double ro = ri + reactor.setup[current_section_num].get_size();
            const double s = sample_dist(reactor, current_section);
            const double theta = acos(1 - 2*random(reactor.rd));
            const double phi = 2*M_PI*random(reactor.rd);
            x += s*cos(phi)*sin(theta);
            y += s*sin(phi)*sin(theta);
            z += s*cos(theta);
            r = sqrt(x*x + y*y + z*z);
            // temporary
            // std::cout << "I have then moved s = " << s << "\n";
            // std::cout << "theta = " << theta << "\n";
            // std::cout << "phi = " << phi << "\n";
            // print_coords();
            if(r >= ro){
                if(r >= reactor.get_size()){
                    // we have transmitted
                    reactor.transmitted++;
                    is_alive = false;
                    break;
                }
                // std::cout << "I have traversed further out\n";
                // print_coords();
                // traversed out through ro and we have to backtrack to boundary
                // we will recalculate s such that the neutron traverses right onto the boundary
                // quadratic components
                x -= s*cos(phi)*sin(theta);
                y -= s*sin(phi)*sin(theta);
                z -= s*cos(theta);
                // r = sqrt(x*x + y*y + z*z);
                // std::cout << "I now backtrack\n";
                // print_coords();
                double b = 2.0*(x*cos(phi)*sin(theta) +
                                y*sin(phi)*sin(theta) +
                                z*cos(theta));
                double c = (x*x + y*y + z*z - ro*ro);
                double discriminant = b*b - 4.0*c;

                double s_new = (-b + sqrt(discriminant)) / 2.0;
                x += (s_new + 1e-7)*cos(phi)*sin(theta);
                y += (s_new + 1e-7)*sin(phi)*sin(theta);
                z += (s_new + 1e-7)*cos(theta);
                // we nudge the particle slightly past the boundary so that we get cross section data in the next section
                r = sqrt(x*x + y*y + z*z);
                // std::cout << "Here are my new coords: r = " << r << "\n";
                // print_coords();
            }else if(r < ri){
                // std::cout << "I have traversed closer in\n";
                // print_coords();
                // traversed in through ri and we have to backtrack to boundary
                // we will recalculate s such that the neutron traverses right onto the boundary
                // quadratic components
                x -= s*cos(phi)*sin(theta);
                y -= s*sin(phi)*sin(theta);
                z -= s*cos(theta);
                // r = sqrt(x*x + y*y + z*z);
                // std::cout << "I now backtrack\n";
                // print_coords();
                double b = 2.0*(x*cos(phi)*sin(theta) +
                                y*sin(phi)*sin(theta) +
                                z*cos(theta));
                double c = (x*x + y*y + z*z - ri*ri);
                double discriminant = b*b - 4.0*c;

                double s_new = (-b - sqrt(discriminant)) / 2.0;
                x += (s_new - 1e-7)*cos(phi)*sin(theta);
                y += (s_new - 1e-7)*sin(phi)*sin(theta);
                z += (s_new - 1e-7)*cos(theta);
                // we nudge the particle slightly past the boundary so that we get cross section data in the next section
                r = sqrt(x*x + y*y + z*z);
                // std::cout << "Here are my new coords: r = " << r << "\n";
                // print_coords();
            }else{
                // else we sample a collision within the current section
                int collision = sample_collision(reactor, current_section);
                if(collision == 0){
                    // absorbed
                    reactor.absorbed++;
                    is_alive = false;
                }else if(collision == 1){
                    // fissioned
                    reactor.fissioned++;
                    is_alive = false;
                }else{
                    // scattered
                    continue;
                }
            }
        }
    }
    else if(reactor.get_geometry() == 0){
        // slab reactor
        while(is_alive){
            int current_section_num = reactor.get_location(r);
            Section current_section = reactor.setup[current_section_num];
            // to determine if the neutron has crossed a boundary or not
            double ri = 0;
            for(int i = 0; i < current_section_num; i++){
                ri += reactor.setup[i].get_size();
                // ro += reactor.setup[i].get_size();
            }
            double ro = ri + reactor.setup[current_section_num].get_size();
            const double s = sample_dist(reactor, current_section);
            const double mu = 1 - 2*random(reactor.rd);
            // std::cout << s*mu << "\n";
            r += s*mu;
            if(r >= ro){
                if(r >= reactor.get_size()){
                    // we have transmitted
                    // std::cout << "I have transmitted out\n";
                    // print_coords();
                    reactor.transmitted++;
                    is_alive = false;
                    break;
                }
                // std::cout << "I have traversed further out\n";
                // print_coords();
                // traversed out through ro and we have to backtrack to boundary
                // we will recalculate s such that the neutron traverses right onto the boundary
                // quadratic components
                // std::cout << s*mu << "\n";
                r -= s*mu;
                // std::cout << mu << "\n";
                // r = sqrt(x*x + y*y + z*z);
                // std::cout << "I now backtrack\n";
                // print_coords();
                double s_new = ro - r; 
                r += s_new + 1e-7;
                // we nudge the particle slightly past the boundary so that we get cross section data in the next section
  
                // std::cout << "Here are my new coords: r = " << r << "\n";
                // print_coords();
            }else if(r < ri){
                if(r < 0.0){
                    // we have reflected
                    // std::cout << "I have reflected back out\n";
                    // print_coords();
                    reactor.reflected++;
                    is_alive = false;
                    break;
                }
                // std::cout << "I have traversed closer in\n";
                // print_coords();
                // traversed in through ri and we have to backtrack to boundary
                // we will recalculate s such that the neutron traverses right onto the boundary
                // quadratic components
                r -= s*mu;
                // r = sqrt(x*x + y*y + z*z);
                // std::cout << "I now backtrack\n";
                // print_coords();
                double s_new = r - ri; 
                r -= (s_new + 1e-7);
                // we nudge the particle slightly past the boundary so that we get cross section data in the next section

                // std::cout << "Here are my new coords: r = " << r << "\n";
                // print_coords();
            }else{
                // else we sample a collision within the current section
                int collision = sample_collision(reactor, current_section);
                if(collision == 0){
                    // absorbed
                    reactor.absorbed++;
                    is_alive = false;
                }else if(collision == 1){
                    // fissioned
                    reactor.fissioned++;
                    is_alive = false;
                }else{
                    // scattered
                    continue;
                }
            }
        }
    }
}
void neutron::initialize(Reactor &reactor, int core_section_num){
    std::uniform_real_distribution<double> random{0.0, 1.0};
    Section core_section = reactor.setup[core_section_num];
    double ri = 0;
    for(int i = 0; i < core_section_num; i++){
        ri += reactor.setup[i].get_size();
        // ro += reactor.setup[i].get_size();
    }
    double ro = ri + reactor.setup[core_section_num].get_size();
    // temporary
    // std::cout << "ri = " << ri << "\n";
    // std::cout << "ro = " << ro << "\n";
    if(reactor.get_geometry() == 1){
        // spherical geometry
        double sampled_r = cbrt((pow(ro, 3) - pow(ri, 3))*random(reactor.rd) + pow(ri, 3));
        double sampled_theta = acos(1 - 2*random(reactor.rd));
        double sampled_phi = 2*M_PI*random(reactor.rd);

        x = sampled_r*cos(sampled_phi)*sin(sampled_theta);
        y = sampled_r*sin(sampled_phi)*sin(sampled_theta);
        z = sampled_r*cos(sampled_theta);
        r = sampled_r;
        
        // temporary
        // for(int i = 0; i < 10; i++){
        //     double sampled_r = cbrt((pow(ro, 3) - pow(ri, 3))*random(reactor.rd) + pow(ri, 3));
        //     std::cout << sampled_r << " ";
        // }
        // std::cout << "\n";
        // std::cout << "I have initialized my position at r = " << r << "\n";
    }else if(reactor.get_geometry() == 0){
        // slab geometry
        double sampled_r = ri + random(reactor.rd)*(ro - ri);
        r = sampled_r;
    }
}
int neutron::sample_collision(Reactor &reactor, const Section &current_section){
    // add a function that can sample a neutron collision
    std::uniform_real_distribution random{0.0, 1.0};
    double Sig_a = current_section.get_Sig_a();
    double Sig_s = current_section.get_Sig_s();
    double Sig_f = current_section.get_Sig_f();
    double Sig_t = current_section.get_Sig_t();

    double eta = random(reactor.rd);
    if(eta <= Sig_a/Sig_t){
        // we absorb
        // reactor.absorbed++;
        return 0;
    }else if(Sig_a/Sig_t < eta && eta <= (Sig_a + Sig_f)/Sig_t){
        // we fission
        // reactor.fissioned++;
        return 1;
    }else{
        // we scatter
        return 2;
    }
    return 0;
}
struct SimulationResult{
    int transmitted;
    int absorbed;
    int fissioned;
    int reflected;
};
SimulationResult run_simulation(Reactor &reactor, int num_neutrons, int core_num){
    reactor.reset_tallies();
    for(int i = 0; i < num_neutrons; i++){
        neutron n;
        n.initialize(reactor, core_num);
        n.traverse(reactor);
    }
    if(reactor.get_geometry() == 1){
        return {reactor.transmitted, reactor.absorbed, reactor.fissioned, 0};
    }else if(reactor.get_geometry() == 0){
        return {reactor.transmitted, reactor.absorbed, reactor.fissioned, reactor.reflected};
    }
    return {-1, -1, -1, -1};
}
int main(){
    using std::cout;
    Reactor my_reactor(1);
    Section plutonium(0.33129, 0.10280, 0.09013, 2.3);
    Section iron(0.031311, 5.0724e-4, 0.0, 1250.0);
    my_reactor.add_section(plutonium);
    my_reactor.add_section(iron);

    // Reactor my_reactor(0);
    // Section s1(0.09, 0.01, 0.0, 0.10);
    // Section s2(9.9, 0.1, 0.0, 0.10);
    // Section s3(90.0, 10.0, 0.0, 0.10);
    // my_reactor.add_section(s1);
    // my_reactor.add_section(s2);
    // my_reactor.add_section(s3);

    // neutron n0 = my_reactor.n0;
    // n0.initialize(my_reactor, 0);
    // n0.print_coords();
    // n0.traverse(my_reactor);
    // n0.print_coords();

    int num_neutrons = 5000;
    SimulationResult result = run_simulation(my_reactor, num_neutrons, 0);
    // std::cout << "Fissions: " << result.fissioned << "\n";
    
    auto [t, a, f, r] = result;

    std::cout << std::setprecision(6);
    cout << "Transmitted: " << (double)t/num_neutrons << "\n";
    cout << "Absorbed: " << (double)a/num_neutrons << "\n";
    cout << "Fissioned: " << (double)f/num_neutrons << "\n";
    cout << "Reflected: " << (double)r/num_neutrons << "\n";

    int num_tests = 250;
    std::ofstream my_file("toymc_data.csv");
    my_file << "transmitted,absorbed,fissioned\n";
    for(int i = 0; i < num_tests; i++){
        SimulationResult result = run_simulation(my_reactor, num_neutrons, 0);
        auto [t, a, f, r] = result;
        my_file << (double)t/num_neutrons << "," << (double)a/num_neutrons << "," << (double)f/num_neutrons << "\n";
    }

    return 0;
}