NEGATIVE_TRIGGER_LABEL = 'O'
NEGATIVE_ARGUMENT_LABEL = 'no_arg'
NEGATIVE_ENTITY_LABEL = 'O'
SD4M_RELATION_TYPES = ['Accident', 'CanceledRoute', 'CanceledStop', 'Delay',
                       'Obstruction', 'RailReplacementService', 'TrafficJam',
                       NEGATIVE_TRIGGER_LABEL]
SDW_RELATION_TYPES = ["Disaster", "Insolvency", "Layoffs", "Merger", "SpinOff", "Strike",
                      "CompanyProvidesProduct", "CompanyUsesProduct", "CompanyTurnover",
                      "CompanyRelationship", "CompanyFacility", "CompanyIndustry",
                      "CompanyHeadquarters", "CompanyWebsite", "CompanyWikipediaSite",
                      "CompanyNumEmployees", "CompanyCustomer", "CompanyProject",
                      "CompanyFoundation", "CompanyTermination", "CompanyFinancialEvent",
                      NEGATIVE_TRIGGER_LABEL]
ROLE_LABELS = ['location', 'delay', 'direction',
               'start_loc', 'end_loc',
               'start_date', 'end_date', 'cause',
               'jam_length', 'route', NEGATIVE_ARGUMENT_LABEL]
SDW_ROLE_LABELS = ['buyer', 'acquired', 'seller', 'date', 'price', 'company',
                   'location', 'num-laid-off', 'old', 'new', 'organization',
                   'person', 'position', 'from', 'to', 'parent', 'child',
                   'num_striking', 'striker', 'union', NEGATIVE_ARGUMENT_LABEL]
from eventx.dataset_readers import *
from eventx.models import *
from eventx.predictors import *
from eventx.util import *
