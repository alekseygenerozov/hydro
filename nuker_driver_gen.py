#!/usr/bin/env python

import galaxy
import dill
from ConfigParser import SafeConfigParser
import ConfigParser
import argparse

import ast
import sys
import re

from bash_command import bash_command as bc


def config_items_section(config,sec):
		'''Return items in a section w/o the defaults'''
		try:
			items=config.options(sec)
		except ConfigParser.NoSectionError:
			return []

		defaults=config.defaults().keys()
		items=list(set(items)-set(defaults))
		return items

def config_parse_section(config, sec, items=None):
	'''Parse parameters in a given section of a ConfigParser object'''
	if not items:
		items=config_items_section(config, sec)
	param_dict={}
	for name in items:
		try:
			param_dict[name]=ast.literal_eval(config.get(sec,name))
		except ValueError,err:
			param_dict[name]=config.get(sec,name)
		except ConfigParser.NoOptionError:
			continue
		except ConfigParser.NoSectionError:
			break

	return param_dict

def ast_literal_eval_safe(s):
	try:
		return ast.literal_eval(s)
	except ValueError,err:
		return s

class LazyCallable(object):
	def __init__(self, name):
		self.n, self.f = name, None
	def __call__(self, *a, **k):
		if self.f is None:
			modn, funcn = self.n.rsplit('.', 1)
			if modn not in sys.modules:
				__import__(modn)
			self.f = getattr(sys.modules[modn],funcn)
		return self.f(*a, **k)

class Driver(object):
	'''Driver to set up solutions for Nuker galaxies'''
	def __init__(self, config_file):
		self.config_file=config_file
		self.__parse_config() 
		if self.name=='quataert':
			if self.model:
				self.gal=galaxy.Galaxy.from_dir(loc=self.model, **self.grid_params_dict)
			else:
				self.gal=galaxy.Galaxy(init=self.grid_params_dict)
		elif self.name=='pow':
			if self.model:
				self.gal=galaxy.PowGalaxy.from_dir(loc=self.model, **self.grid_params_dict)
			else:
				self.gal=galaxy.PowGalaxy(init=self.grid_params_dict)
		elif re.match('.*_Extend', self.name):
			self.name=self.name.replace('_Extend', '')
			if self.model:
				self.gal=galaxy.NukerGalaxyExtend.from_dir(args=[self.name, self.gdata_dict], loc=self.model, **self.grid_params_dict)
			else:
				self.gal=galaxy.NukerGalaxyExtend(self.name, gdata=self.gdata_dict, init=self.grid_params_dict)
		else:
			if self.model:
				self.gal=galaxy.NukerGalaxy.from_dir(args=[self.name, self.gdata_dict], loc=self.model, **self.grid_params_dict)
			else:
				self.gal=galaxy.NukerGalaxy(self.name, gdata=self.gdata_dict, init=self.grid_params_dict)

		for param in self.model_params_dict:
			self.gal.set_param(param,self.model_params_dict[param])
		for param in self.user_params_dict:
			self.gal.set_param(param,self.user_params_dict[param])

	def __parse_config(self):
		self.config=SafeConfigParser(defaults={'model_params':'True'})
		self.config.optionxform=str
		try:
			self.config.read(self.config_file)
		except ConfigParser.ParsingError, err:
			print 'Could not parse:', err 
		self.name=ast_literal_eval_safe(self.config.get('name','name'))
		try:
			self.model=ast_literal_eval_safe(self.config.get('model','model'))
			self.model_params=self.config.getboolean('model','model_params')
		except (ConfigParser.NoOptionError,ConfigParser.NoSectionError):
			self.model=None
			self.model_params=None

		self.model_params_dict={}
		if self.model_params and self.model:
			self.model_params_dict=dill.load(open(self.model+'/non_standard.p','rb'))

		self.__parse_config_params()
		self.__parse_config_grid()
		self.__parse_config_adjust()
		self.__parse_config_gdata()

	def __parse_config_params(self):
		self.user_params_dict=config_parse_section(self.config, 'params')

	def __parse_config_grid(self):
		if self.model:
			grid_params_names=['rmin', 'rmax', 'logr', 'length', 'rescale', 'index','extrap']
		else:
			grid_params_names=['rmin', 'rmax', 'logr', 'length', 'f_initial', 'func_params']
		self.grid_params_dict={}
		self.grid_params_dict=config_parse_section(self.config, 'grid', grid_params_names)
		if 'f_initial' in self.grid_params_dict:
			self.grid_params_dict['f_initial']=LazyCallable(self.grid_params_dict['f_initial'])

	def __parse_config_adjust(self):
		self.adjust_params_dict=config_parse_section(self.config,'adjust')

	def __parse_config_gdata(self):
		try:
			self.gdata_dict=config_parse_section(self.config, 'gdata')['gdata']
		except KeyError:
			self.gdata_dict=config_parse_section(self.config, 'gdata')
	def solve(self):
		'''Find solution for given galaxy'''
		bc('cp '+self.config_file+' '+self.gal.outdir)
		for param in self.adjust_params_dict:
			self.gal.solve_adjust(5.*self.gal.tcross, param, self.adjust_params_dict[param])
		try:
			time=self.config.getfloat('time','time')
		except:
			time=None
		if time:
			self.gal.solve(time*self.gal.tcross)
		else:
			self.gal.solve()

def main():
	parser=argparse.ArgumentParser(
		description='Driver for generating Nuker solutions')
	parser.add_argument('config', nargs=1,
		help='File containing config information for our run.')

	args=parser.parse_args()
	config_file=args.config[0]

	driver=Driver(config_file)
	driver.solve()


if __name__ == '__main__':
	main()








