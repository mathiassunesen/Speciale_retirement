from Model import RetirementModelClass
updpar = dict()
updpar["Na"] = 500
model = RetirementModelClass(name="baseline",solmethod="egm",**updpar)
model.test()
