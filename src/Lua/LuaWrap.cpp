
// Lua and toLua Wrapper
////////////////////////////////////////////////////////////////////////////////
/********************************************************
* @file: luaWrap.cpp
* @ ver: 1.0
* @date: 11/03/2012
* @ mod: xx/xx/2012
********************************************************/

// Headers
////////////////////////////////////////////////////////////////////////////////
#include "LuaWrap.h"
//standard
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
//lua - tolua
#include <lua.hpp>


// initializations
LuaWrap* LuaWrap::m_luaWrap = NULL;

// constructor
////////////////////////////////////////////////////////////////////////////////
LuaWrap::LuaWrap()
: m_luastate ( NULL ) // inicializa como nulo o lua state
, status     ( NULL ) // inicializa como nulo o status
{
}

// destructor
////////////////////////////////////////////////////////////////////////////////
LuaWrap::~LuaWrap() {
  lua_close(m_luastate);
  delete m_luaWrap;
}

// gets luaWrap single instance
////////////////////////////////////////////////////////////////////////////////
LuaWrap& LuaWrap::instance() {
  return m_luaWrap ? *m_luaWrap : *(m_luaWrap = new LuaWrap());
  if (!m_luaWrap) {
    m_luaWrap = new LuaWrap();
  }
  return *m_luaWrap;
}

// initializes luaWrap (lua and tolua libs)
////////////////////////////////////////////////////////////////////////////////
void LuaWrap::init() {
  m_luastate = luaL_newstate();   /* opens Lua */
  luaL_openlibs(m_luastate);      /* auxiliary Lua libs. */
  //luaopen_commands(m_luastate);
  //tolua_commands_open(m_luastate);
}

// gets current lua state
////////////////////////////////////////////////////////////////////////////////
lua_State* LuaWrap::getLuaState() {
  return m_luastate;
}

// pushes a double to the stack
////////////////////////////////////////////////////////////////////////////////
void LuaWrap::pushnumber(double n) {
  lua_pushnumber(m_luastate, n);
}

// pushes a int to the stack
////////////////////////////////////////////////////////////////////////////////
void LuaWrap::pushint(int n) {
  lua_pushinteger(m_luastate, n);
}

// pushes a int to the stack
////////////////////////////////////////////////////////////////////////////////
void LuaWrap::pushstring(const char* s) {
  lua_pushstring(m_luastate, s);
}

// pushes a int to the stack
////////////////////////////////////////////////////////////////////////////////
void LuaWrap::pushnil() {
  lua_pushnil(m_luastate);
}

// pushes a userdata to stack
////////////////////////////////////////////////////////////////////////////////
void LuaWrap::pushuserdata(void* p) {
  lua_pushlightuserdata(m_luastate, p);
}

// pops top element of stack
////////////////////////////////////////////////////////////////////////////////
void LuaWrap::pop() {
  lua_pop(m_luastate, -1);
}

// returns poped integer from stack
////////////////////////////////////////////////////////////////////////////////
int LuaWrap::popint() {
  int n = lua_tointeger(m_luastate, -1);
  lua_pop(m_luastate, 1);

  return n;
}

// returns popped double from stack
////////////////////////////////////////////////////////////////////////////////
double LuaWrap::popnumber() {
  double num = lua_tonumber (m_luastate, -1);
  lua_pop(m_luastate, 1);

  return num;
}

// returns popped string from stack
////////////////////////////////////////////////////////////////////////////////
const char* LuaWrap::popstring() {
  const char* string = lua_tostring(m_luastate, -1);
  lua_pop(m_luastate, 1);

  return string;
}

// pops a userdata off stack
////////////////////////////////////////////////////////////////////////////////
void* LuaWrap::popuserdata() {
  void* vp = lua_touserdata(m_luastate, -1);
  lua_pop(m_luastate, 1);

  return vp;
}

// dumps stack contents simply as strings
////////////////////////////////////////////////////////////////////////////////
void LuaWrap::stackDump() {
  int n = lua_gettop(m_luastate);
  printf("Number of Elements on Stack: %i\n", n);
  for( int i=1; i<=n; i++ )
  {
    printf("Stack[%i]: %s\n", i, lua_tostring(m_luastate, -(i)));
  }
}

// gets a global variable
////////////////////////////////////////////////////////////////////////////////
void LuaWrap::getglobal(const char* name) {
  lua_getglobal(m_luastate, name);
}

// gets a global variable "name"
////////////////////////////////////////////////////////////////////////////////
void LuaWrap::setglobal(const char* name) {
  lua_setglobal(m_luastate, name);
  lua_pop(m_luastate, 1);
}

// executes a lua file
////////////////////////////////////////////////////////////////////////////////
int LuaWrap::dofile(const char* filename) {
  int ret = luaL_dofile(m_luastate, filename);
  if ( ret == 1 )
    printf("Error running luaWrap::dofile doing file: %s\nerror: %s\n",
           filename, lua_tostring(m_luastate, -1));
  return ret;
}

// calls a pushed functions given number of arguments and results
////////////////////////////////////////////////////////////////////////////////
void LuaWrap::callfunction( int nargs, int nresults ) {
  if (lua_pcall(m_luastate, nargs, nresults, 0) != 0)
    printf("Error running function %s: %s",
           lua_tostring(m_luastate, -(nargs+1)),lua_tostring(m_luastate, -1));
}

// registers a function with lua
////////////////////////////////////////////////////////////////////////////////
void LuaWrap::registerfunc( char* funcname, void (*f)(void)) {
  lua_register(m_luastate, funcname, (lua_CFunction)f);
}

// returns 1 if index is function, 0 otherwise
////////////////////////////////////////////////////////////////////////////////
int LuaWrap::isfunc( int index ) {
  return lua_isfunction( m_luastate, index );
}

// returns 1 if index is nil, 0 otherwise
////////////////////////////////////////////////////////////////////////////////
int LuaWrap::isnil( int index ) {
  int ret = lua_isnil( m_luastate, index );
  return ret;
}

// returns lua error integer
////////////////////////////////////////////////////////////////////////////////
int LuaWrap::getError( ) {
  char* err = NULL;
  int ret = luaL_error( m_luastate, err );
  return ret;
}
