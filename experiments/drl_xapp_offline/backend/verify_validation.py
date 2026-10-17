"""Compile the exact bridge validator in a minimal harness and test its boundaries."""
from pathlib import Path
import json
import subprocess
import tempfile


def main():
    root = Path(__file__).resolve().parent
    header = (root / "drl-lockstep-control.h").read_text()
    start = header.index("    static void Validate(")
    end = header.index("\n    /**", start)
    method = header[start:end]
    program = r'''
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <iostream>
static void Require(bool valid, const std::string& message) {
    if (!valid) throw std::runtime_error(message);
}
''' + method + r'''
int main() {
    int checks = 0;
    auto check = [&](const std::string& key, double value, bool shouldPass) {
        bool passed = true;
        try { Validate(key, value); } catch (const std::runtime_error&) { passed = false; }
        if (passed != shouldPass) {
            std::cerr << "Unexpected validation: " << key << "=" << value << "\n";
            std::exit(1);
        }
        ++checks;
    };
    for (double value : {5.,6.,7.,8.,9.}) check("cio_5g",value,true);
    for (const std::string key : {"cio_4g1","cio_4g2"}) {
        for (double value : {-2.,-1.,0.,1.,2.}) check(key,value,true);
        for (double value : {-3.,3.,0.5}) check(key,value,false);
    }
    for (double value : {128.,256.,340.,512.,720.}) check("ttt_ms",value,true);
    for (double value : {40.,64.,320.,640.,1024.,340.5}) check("ttt_ms",value,false);
    for (int i=1;i<=10;++i) check("hys_db",i/2.0,true);
    for (double value : {0.,0.25,1.25,5.5,6.}) check("hys_db",value,false);
    for (double value : {4.,10.,5.5}) check("cio_5g",value,false);
    for (const std::string key : {"cio_5g","cio_4g1","cio_4g2","ttt_ms","hys_db"}) {
        check(key,std::numeric_limits<double>::quiet_NaN(),false);
        check(key,std::numeric_limits<double>::infinity(),false);
    }
    check("txp_5g",43,false);
    check("ret_5g",9,false);
    std::cout << checks << "\n";
}
'''
    with tempfile.TemporaryDirectory(prefix="validate_", dir=root) as temporary:
        source = Path(temporary) / "validate.cc"
        binary = Path(temporary) / "validate"
        source.write_text(program)
        subprocess.run(["g++", "-std=c++17", "-O0", str(source), "-o", str(binary)], check=True)
        result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    summary = {"passed": True, "validation_cases": int(result.stdout.strip()),
               "scope": "Exact C++ bridge Validate function; no ns3 simulation launched"}
    (root / "validation_verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
