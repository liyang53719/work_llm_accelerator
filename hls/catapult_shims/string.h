#ifndef TVM_WORK_LLM_ACCELERATOR_CATAPULT_SHIMS_STRING_H
#define TVM_WORK_LLM_ACCELERATOR_CATAPULT_SHIMS_STRING_H

namespace std {

class string {
public:
  string() {}
  string(const char*) {}
  string(const string&) {}

  string& operator=(const char*) { return *this; }
  string& operator=(const string&) { return *this; }
  string& operator+=(const char*) { return *this; }
  string& operator+=(const string&) { return *this; }
  string& operator+=(char) { return *this; }
};

inline string operator+(const string&, const string&) {
  return string();
}

inline string operator+(const string&, const char*) {
  return string();
}

inline string operator+(const char*, const string&) {
  return string();
}

inline string operator+(const string&, char) {
  return string();
}

}  // namespace std

#endif