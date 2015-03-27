#pragma once

#include <assert.h>

enum Location{LocCPU, LocGPU};

class Loc{
  protected:
    Location loc;
    
  public:
    virtual Location getLocation() const{
      return loc;
    }
};
class CPULocation:public Loc{
  
  public:
    CPU(){
      loc = LocCPU;
    }
};

class GPULocation:public Loc{
  
  public:
    GPU(){
      loc = LocGPU;
    }
};
/*
class Bond{

};

class BondCPU:public CPU, public Bond{

};

class BondGPU:public GPU, public Bond{

};
void test(){
  BondCPU c;
  assert(LocCPU == c.getLocation());
  
  BondGPU g;
  assert(LocGPU == g.getLocation());
}

int main(int argc, char *argv[]){
  test();
}
*/

