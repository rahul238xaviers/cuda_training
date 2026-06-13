#include <iostream>


void moidify_by_value(float val){
    val *= 2.0f;
    
}

void modify_by_pointer(float* ptr){

    if(ptr != nullptr){
        *ptr *= 2.0f;
    }
}

void modify_by_reference(float& val){
    val *= 2.0f;
}

int main(){

    float val = 10.0f;

    std::cout<<"The value of val variable initially " << val << std::endl;
    moidify_by_value(val);
    std::cout<<"The value of val variable after calling moidify_by_value function " << val << std::endl;

    modify_by_pointer(&val);
    std::cout<<"The value of val variable after calling modify_by_pointer function " << val << std::endl;

    modify_by_reference(val);
    std::cout<<"The value of val variable after calling modify_by_reference function " << val << std::endl;

    return 0;
}