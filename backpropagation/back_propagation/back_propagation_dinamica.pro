QT += core
QT -= gui

CONFIG += c++11

TARGET = back_propagation_dinamica
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    cbackprop.cpp

HEADERS += \
    cbackprop.h
