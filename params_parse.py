from ConfigParser import SafeConfigParser
import ast

class CaseConfigParser(SafeConfigParser):
	def __init__(self):
		SafeConfigParser.__init__(self)
		self.optionxform = str

def params_parse(conf_file):
	config = CaseConfigParser()
	params_dict={}
	param_names=['Re', 'Re_s', 'visc_scheme', 'bdry', 'sinterval', 'tinterval', 'eps', 'sigma_heating', 'isot']

	try:
		f=open(conf_file,'r')
		config.readfp(f)
	except: 
		return {}
	for name in param_names:
		try:
			params_dict[name]=ast.literal_eval(config.get('params',name))
		except:
			pass

	return params_dict