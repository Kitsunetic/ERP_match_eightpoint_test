#ifndef _PATHKIT_H_
#define _PATHKIT_H_

#define MAX(a,b)  ((a) < (b) ? (b) : (a))

#include <iostream>
#include <sstream>
#include <vector>

using namespace std;


class PathKit {
    public:
    void split_ext(string path, string &name, string &ext) {
        int d = path.find('.');
        name = path.substr(0, d);
        ext = path.substr(d);
    }

    void replace_string(string &path, char src, char dst) {
        int p;
        while(true) {
            p = path.find(src);
            if(p < 0) break;
            
            path[p] = dst;
        }
    }

    string dirname(string path) {
        int P = max(0, path.rfind('/'), path.rfind('\\'));
        return path.substr(0, P); 
    }

    string basename(string path) {
        int P = max(path.rfind('/'), path.rfind('\\'));
        if(P == -1) return path;
        return path.substr(P+1, path.length()-P-1);
    }
    
    private:
    int max(int a, int b) {
        if(a > b) return a;
        return b;
    }
    
    int max(int a, int b, int c) {
        if(a > b) {
            if(c > a) return c;
            else return a;
        }
        else if(c > b) return c;
        else return b;
    }
};

#endif
