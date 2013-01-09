
// Lua and toLua Wrapper Header File
////////////////////////////////////////////////////////////////////////////////
/********************************************************
* @file: luaWrap.h
* @ ver: 1.1
* @date: 11/03/2012
* @ mod: 18/06/2012
********************************************************/

#ifndef luaWrap_H
#define luaWrap_H

// Headers
////////////////////////////////////////////////////////////////////////////////
#include <lua.hpp>

// Singleton Implementation
////////////////////////////////////////////////////////////////////////////////

// macro for easy singleton access
#define WLua LuaWrap::instance()

// Class: luaWrap - singleton for management of lua state and calls
class LuaWrap
{

public:

  // @brief accesses the singleton object.
  static LuaWrap& instance();


protected:

  // @brief default constructor.
  LuaWrap();

  // @brief destructor.
  virtual ~LuaWrap();


public:

  // lua handling functions.
  void       init( );
  lua_State* getLuaState();

  // stack manipulation functions
  void        pushnumber( double n );
  void        pushint(int n);
  void        pushstring(const char* s);
  void        pushnil();
  void        pushuserdata(void* p);
  void        pop();
  int         popint();
  double      popnumber();
  const char* popstring();
  // warning: be careful to use type casting correctly!
  void*       popuserdata();
  void        stackDump();

  void getglobal(const char* name);
  void setglobal(const char* name);
  int  dofile(const char* filename);
  void callfunction( int nargs, int nresults );
  void registerfunc( char* funcname, void (*f)(void));
  int  isfunc( int index );
  int  isnil( int index );
  int  getError( );

private:
  // Variables:
  static LuaWrap* m_luaWrap; // pointer to singleton
  lua_State*      m_luastate;
  char*           status;
};

#endif
