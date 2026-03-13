#ifndef TVM_WORK_LLM_ACCELERATOR_CATAPULT_SHIMS_OSTREAM_H
#define TVM_WORK_LLM_ACCELERATOR_CATAPULT_SHIMS_OSTREAM_H

namespace std {

class ios_base {
public:
  enum fmtflags {
    dec = 0,
    hex = 1,
    oct = 2
  };
};

class ostream {
public:
  ostream& operator<<(const char*) { return *this; }
  ostream& operator<<(char) { return *this; }
  ostream& operator<<(signed char) { return *this; }
  ostream& operator<<(unsigned char) { return *this; }
  ostream& operator<<(short) { return *this; }
  ostream& operator<<(unsigned short) { return *this; }
  ostream& operator<<(int) { return *this; }
  ostream& operator<<(unsigned int) { return *this; }
  ostream& operator<<(long) { return *this; }
  ostream& operator<<(unsigned long) { return *this; }
  ostream& operator<<(long long) { return *this; }
  ostream& operator<<(unsigned long long) { return *this; }
  ostream& operator<<(bool) { return *this; }
  ostream& operator<<(float) { return *this; }
  ostream& operator<<(double) { return *this; }
  ostream& operator<<(long double) { return *this; }
  ostream& operator<<(ios_base::fmtflags) { return *this; }
  ostream& setf(ios_base::fmtflags) { return *this; }
};

extern ostream cout;
extern ostream cerr;
extern const char endl;

}  // namespace std

#endif