// Copyright 2015 Georg-August-Universität Göttingen, Germany
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

package de.ugoe.cs.cpdp.dataprocessing;

import de.ugoe.cs.cpdp.IParameterizable;
import de.ugoe.cs.cpdp.versions.SoftwareVersion;

/**
 * A data processing strategy that is applied to the software version of the test data and a a single software version as training data.
 * 
 * @author Steffen Herbold
 */
public interface IProcessesingStrategy extends IParameterizable {

    /**
     * Applies the processing strategy.
     * 
     * @param testversion
     *            version of the test data
     * @param trainversion
     *            version of the training data
     */
    void apply(SoftwareVersion testversion, SoftwareVersion trainversion);
}
